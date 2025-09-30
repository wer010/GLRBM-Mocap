from typing import Tuple, TYPE_CHECKING, Union

import torch
from pytorch3d.common.workaround import symeig3x3
from pytorch3d.ops.knn import  knn_points
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from pytorch3d.transforms import matrix_to_axis_angle
import torch.nn.functional as F

def estimate_lcs_with_curv(
    idx,
    p2,
    K
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 1. build local coordinate system with K nearest points
    num_p1 = idx.shape[1]
    idx = idx.flatten()
    if p2.dim()==2:
        p2 = p2[None,...]
    pcl = p2[:,idx,:].reshape(-1, num_p1, K, 3)
    pcl_mean = pcl.sum(-2, keepdim = True)/K
    pcl_centered = pcl - pcl_mean

    cov = torch.matmul(pcl_centered.transpose(-1,-2), pcl_centered)
    curvatures, local_coord_frames = symeig3x3(cov, eigenvectors=True)

    # 3. make sure the local coordinate system is right-handed
    n = _disambiguate_vector_directions(
        pcl_centered[:,:,0,:],  local_coord_frames[:, :, :, 0]
    )
    # disambiguate the main curvature
    z = _disambiguate_vector_directions(
        pcl_centered[:,:,0,:],  local_coord_frames[:, :, :, 2]
    )

    y = torch.cross(z, n, dim=2)
    local_coord_frames = torch.stack((n, y, z), dim=3)

    # 4. construct se3 matrices
    ret = torch.zeros(local_coord_frames.shape[0], local_coord_frames.shape[1], 4, 4, dtype=local_coord_frames.dtype, device=local_coord_frames.device)
    ret[:, :, :3, :3] = local_coord_frames
    ret[:, :, :3, 3] = pcl[:, :, 0, :]
    ret[:, :, 3, 3] = 1.0

    return ret


def estimate_lcs_with_faces(vid, fid, vertices, faces):
    '''
    :param vid: shape [B, N] the index of vertices which will be calculated the LCS
    :param fid: shape [B, N, 9] the triangle index of vertices, each vertex of smpl has most 9 triangle and -1 is padded
    :param vertices: shape [B, 6890, 3] the position of smpl vertices, use vid to obtain the position
    :param faces: shape [13776, 3] all the faces of smpl, restore the vertice indexes of triangles
    :return: lcs: shape [B, N, 4, 4] the lcs (SE(3) matrix) of all vertices
    '''
    assert vertices.dim()==3
    device = vertices.device
    batch = fid.shape[0]
    n_face_per_vertex = fid.shape[-1]
    n_dim = vertices.shape[2]

    f_v = faces[fid, :]
    # v0 = vertices[tri[:,0]]
    # v1 = vertices[tri[:,1]]
    # v2 = vertices[tri[:,2]]
    v0 = torch.take_along_dim(vertices, f_v.reshape(batch, -1, 3)[..., 0:1], dim=1)
    v1 = torch.take_along_dim(vertices, f_v.reshape(batch, -1, 3)[..., 1:2], dim=1)
    v2 = torch.take_along_dim(vertices, f_v.reshape(batch, -1, 3)[..., 2:3], dim=1)

    a = v2 - v1
    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = v0 - v2
    b = b / torch.norm(b, dim=-1, keepdim=True)
    c = v1 - v0
    c = c / torch.norm(c, dim=-1, keepdim=True)

    n = torch.cross(c, -1*b)
    n = n/torch.norm(n, dim=-1, keepdim=True)
    n = n.reshape(batch, -1, n_face_per_vertex, 3)

    # run the cosine and per-row dot product
    angle = torch.zeros_like(v0, dtype=torch.float32).to(device)
    # clip to make sure we don't float error past 1.0
    angle[..., 0] = torch.arccos(torch.clamp((-1*b*c).sum(dim=-1), -1, 1))
    angle[..., 1] = torch.arccos(torch.clamp((-1*c*a).sum(dim=-1), -1, 1))
    # the third angle is just the remaining
    angle[..., 2] = torch.pi - angle[..., 0] - angle[..., 1]

    # a triangle with any zero angles is degenerate
    # so set all of the angles to zero in that case
    angle[(angle < 1e-6).any(dim=2), :] = 0.0
    angle = angle.reshape(batch, -1, n_face_per_vertex, 3)

    mask = (vid[:, :, None, None].expand(-1, -1, n_face_per_vertex, 3) == f_v)
    angle_v = torch.where(mask, angle, torch.zeros_like(angle)).sum(-1)
    angle_v = angle_v/torch.norm(angle_v, dim=-1, keepdim=True)

    nv_v = (n*angle_v[...,None]).sum(-2)
    nv_v = F.normalize(nv_v,dim=-1)

    closest_f = torch.stack([v0,v1,v2], axis = -1).reshape(batch,-1,n_face_per_vertex,3,3)
    closest_f = closest_f[:,:,0,:,:].sum(-1)/3

    vp = torch.take_along_dim(vertices, vid[..., None], dim=1)
    tv_v = closest_f - vp
    z = torch.cross(nv_v, tv_v)
    z = F.normalize(z,dim=-1)
    y = torch.cross(z,nv_v)
    lcs = torch.zeros([z.shape[0],z.shape[1],4,4]).to(device)
    lcs[:,:,0:3, 0] = nv_v
    lcs[:,:,0:3, 1] = y
    lcs[:,:,0:3, 2] = z
    lcs[:,:,0:3, 3] = vp
    lcs[:,:,3, 3] = 1
    return lcs

def invert_se3(T):
    R = T[:, :, :3, :3]  # (B, N, 3, 3)
    t = T[:, :, :3, 3]  # (B, N, 3)

    R_inv = R.transpose(-1, -2)  # (B, N, 3, 3)
    t_inv = -torch.matmul(R_inv, t[..., None]).squeeze(-1)  # (B, N, 3)

    T_inv = torch.zeros_like(T)
    T_inv[:, :, :3, :3] = R_inv
    T_inv[:, :, :3, 3] = t_inv
    T_inv[:, :, 3, 3] = 1.0
    return T_inv

def _disambiguate_vector_directions(df, vecs: torch.Tensor) -> torch.Tensor:

    mask = torch.where(torch.matmul((df[:, :, None, :]), vecs[..., None]) < 0, 1, -1)
    vecs = vecs * mask[..., 0]
    return vecs


def rigid_landmark_transform_torch(a, b):
    """
    PyTorch版本的刚体变换估计

    Args:
        a: tensor of shape (3, N) - 源点集
        b: tensor of shape (3, N) - 目标点集

    Returns:
        R: tensor of shape (3, 3) - 旋转矩阵
        T: tensor of shape (3, 1) - 平移向量

    使得 R @ a + T ≈ b
    基于Arun et al, "Least-squares fitting of two 3-D point sets," 1987.
    """
    assert a.shape[0] == 3, "点集a必须是3xN的形状"
    assert b.shape[0] == 3, "点集b必须是3xN的形状"
    assert a.shape == b.shape, "两个点集形状必须相同"

    # 处理NaN值：用a中对应的值替换b中的NaN
    b = torch.where(torch.isnan(b), a, b)

    # 步骤1: 计算质心
    a_mean = torch.mean(a, dim=1, keepdim=True)  # (3, 1)
    b_mean = torch.mean(b, dim=1, keepdim=True)  # (3, 1)

    # 步骤2: 中心化
    a_centered = a - a_mean  # (3, N)
    b_centered = b - b_mean  # (3, N)

    # 步骤3: 构建协方差矩阵
    C = torch.mm(a_centered, b_centered.T)  # (3, 3)

    # 步骤4: SVD分解
    U, S, Vt = torch.linalg.svd(C, full_matrices=False)
    V = Vt.T  # 转置得到V

    # 步骤5: 计算旋转矩阵
    R = torch.mm(V, U.T)  # (3, 3)

    # 步骤6: 确保右手坐标系 (det(R) > 0)
    if torch.det(R) < 0:
        V_corrected = V.clone()
        V_corrected[:, -1] = -V_corrected[:, -1]  # 翻转最后一列
        R = torch.mm(V_corrected, U.T)

    # 步骤7: 计算平移向量
    T = b_mean - torch.mm(R, a_mean)  # (3, 1)

    return R, T


def rigid_landmark_transform_batch(a, b):
    """
    批量版本的刚体变换估计

    Args:
        a: tensor of shape (B, N, 3) - 批量源点集
        b: tensor of shape (B, N, 3) - 批量目标点集
        b = Ra+T
    Returns:
        R: tensor of shape (B, 3) - 批量轴角表示
        T: tensor of shape (B, 3) - 批量平移向量
    """
    assert a.shape[2] == 3, "点集a的第二维必须是3"
    assert b.shape[2] == 3, "点集b的第二维必须是3"
    assert a.shape == b.shape, "两个点集形状必须相同"

    B = a.shape[0]

    # 处理NaN值
    b = torch.where(torch.isnan(b), a, b)

    # 计算质心
    a_mean = torch.mean(a, dim=1, keepdim=True)  # (B, 1, 3)
    b_mean = torch.mean(b, dim=1, keepdim=True)  # (B, 1, 3)

    # 中心化
    a_centered = a - a_mean  # (B, N, 3)
    b_centered = b - b_mean  # (B, N, 3)

    # 构建协方差矩阵
    C = torch.matmul(a_centered.transpose(-1, -2), b_centered)  # (B, 3, 3)

    # 批量SVD分解
    U, S, Vt = torch.linalg.svd(C, full_matrices=False)
    V = Vt.transpose(-1, -2)  # (B, 3, 3)

    # 计算旋转矩阵
    R = torch.bmm(V, U.transpose(-1, -2))  # (B, 3, 3)

    # 确保右手坐标系
    det_R = torch.det(R)  # (B,)
    flip_mask = det_R < 0

    if flip_mask.any():
        V_corrected = V.clone()
        V_corrected[flip_mask, :, -1] = -V_corrected[flip_mask, :, -1]
        R[flip_mask] = torch.bmm(V_corrected[flip_mask], U[flip_mask].transpose(-1, -2))

    # 计算平移向量
    T = b_mean.transpose(-1,-2) - torch.matmul(R, a_mean.transpose(-1,-2))  # (B, 3, 1)

    return matrix_to_axis_angle(R).squeeze(), T.squeeze()


def axis_angle_distance(aa1, aa2):
    """
    最简洁的轴角距离计算
    基于公式：d = 2*arccos(cos(θ₁/2)cos(θ₂/2) + sin(θ₁/2)sin(θ₂/2)|n₁·n₂|)
    输入aa1,aa2：形状为(N,3)，为两个姿态的轴角表示
    输出dist：形状为(N)，输出两个姿态的测地线距离
    """

    # 角度和轴
    theta1 = torch.norm(aa1, dim=-1)
    theta2 = torch.norm(aa2, dim=-1)

    # 单位轴（处理零向量）
    n1 = aa1 / (theta1.unsqueeze(-1) + 1e-8)
    n2 = aa2 / (theta2.unsqueeze(-1) + 1e-8)

    # 轴点积的绝对值
    axis_dot = torch.sum(n1 * n2, dim=-1)
    axis_dot = torch.clamp(axis_dot, -1.0, 1.0)

    # 半角的三角函数
    cos_half1 = torch.cos(theta1 / 2)
    cos_half2 = torch.cos(theta2 / 2)
    sin_half1 = torch.sin(theta1 / 2)
    sin_half2 = torch.sin(theta2 / 2)

    # 测地距离
    cos_dist = torch.abs(cos_half1 * cos_half2 + sin_half1 * sin_half2 * axis_dot)
    cos_dist = torch.clamp(cos_dist, 0.0, 1.0)  # 距离应该在[0, π/2]

    distance = 2*torch.acos(cos_dist)

    return distance
