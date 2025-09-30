from typing import NewType, Union, Optional
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch
import pyrender
import trimesh
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.point_clouds import PointClouds


Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)


@dataclass
class ModelOutput:
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    v_shaped: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclass
class SMPLOutput(ModelOutput):
    betas: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None


@dataclass
class SMPLHOutput(SMPLOutput):
    left_hand_pose: Optional[Tensor] = None
    right_hand_pose: Optional[Tensor] = None
    transl: Optional[Tensor] = None


@dataclass
class SMPLXOutput(SMPLHOutput):
    expression: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None


@dataclass
class MANOOutput(ModelOutput):
    betas: Optional[Tensor] = None
    hand_pose: Optional[Tensor] = None


@dataclass
class FLAMEOutput(ModelOutput):
    betas: Optional[Tensor] = None
    expression: Optional[Tensor] = None
    jaw_pose: Optional[Tensor] = None
    neck_pose: Optional[Tensor] = None


def find_joint_kin_chain(joint_id, kinematic_tree):
    kin_chain = []
    curr_idx = joint_id
    while curr_idx != -1:
        kin_chain.append(curr_idx)
        curr_idx = kinematic_tree[curr_idx]
    return kin_chain


def to_tensor(
        array: Union[Array, Tensor], dtype=torch.float32
) -> Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)






class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)



def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

def visualize(vertices, faces, extra_point=None, lcs = None, render = False):

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(vertices, faces,
                               vertex_colors=vertex_colors)

    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene = pyrender.Scene()
    scene.add(mesh)

    if extra_point is not None:
        # 定义不同颜色区间和对应的颜色 (RGBA格式)
        color = [
        [0.9, 0.1, 0.1, 1.0],  # 红色 [0:22]
        [0.1, 0.1, 0.9, 1.0],  # 蓝色 [22:25]
        [0.1, 0.9, 0.1, 1.0],  # 绿色 [25:55]
        [0.9, 0.9, 0.1, 1.0] # 黄色 [55:127]
        ]

        # 为每个颜色区间创建球体并添加到场景
        for i, item in enumerate(extra_point):

            # 创建对应颜色的球体
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = color[i]

            # 设置球体的位置变换矩阵
            tfs = np.tile(np.eye(4), (len(item), 1, 1))
            tfs[:, :3, 3] = item

            # 创建Mesh并添加到场景
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

    if lcs is not None:
        add_coordinate_systems_to_scene(scene, lcs, scale=0.1)
    pyrender.Viewer(scene, use_raymond_lighting=True)


def visualize_trimesh(vertices, faces, extra_point=None, lcs=None, show=True, save_path=None):
    # 创建主 mesh
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)

    # 创建场景
    scene = trimesh.Scene()
    scene.add_geometry(tri_mesh)

    # 添加额外点球体
    if extra_point is not None:
        colors = [
            [0.9, 0.1, 0.1, 1.0],  # 红色 [0:22]
            [0.1, 0.1, 0.9, 1.0],  # 蓝色 [22:25]
            [0.1, 0.9, 0.1, 1.0],  # 绿色 [25:55]
            [0.9, 0.9, 0.1, 1.0]  # 黄色 [55:127]
        ]

        for i, group in enumerate(extra_point):
            rgba = (np.array(colors[i]) * 255).astype(np.uint8)
            for point in group:
                sphere = trimesh.creation.uv_sphere(radius=0.005)
                sphere.visual.vertex_colors = rgba
                sphere.apply_translation(point)
                scene.add_geometry(sphere)

    # 添加局部坐标系
    if lcs is not None:
        for tf in lcs:
            tf = np.asarray(tf)
            axis = trimesh.creation.axis(origin_size=0.002, axis_length=0.05)
            axis.apply_transform(tf)
            scene.add_geometry(axis)

    # 显示窗口（可交互旋转）
    if show:
        scene.show()

    # 保存图像（不打开窗口也能保存）
    if save_path is not None:
        png = scene.save_image(resolution=(800, 800), visible=True)
        with open(save_path, 'wb') as f:
            f.write(png)
        print(f"Saved scene image to {save_path}")


def visualize_aitviewer(model_type, full_poses, betas=None, trans=None, extra_points = None):

    smpl_layer = SMPLLayer(model_type=model_type, gender="neutral", device=C.device)
    seq_smpl = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=full_poses[:, 3:],
        poses_root=full_poses[:, 0:3],
        betas=betas,
        trans=trans,
        z_up=True
    )
    v = Viewer()
    v.scene.add(seq_smpl)
    if extra_points is not None:
        # 定义不同颜色区间和对应的颜色 (RGBA格式)
        color = [
        [0.9, 0.1, 0.1, 1.0],  # 红色 [0:22]
        [0.1, 0.1, 0.9, 1.0],  # 蓝色 [22:25]
        [0.1, 0.9, 0.1, 1.0],  # 绿色 [25:55]
        [0.9, 0.9, 0.1, 1.0] # 黄色 [55:127]
        ]

        # 为每个颜色区间创建球体并添加到场景
        for i, item in enumerate(extra_points):
            ptc_amass = PointClouds(item,
                                    color=color[i],
                                    z_up=True)
            v.scene.add(ptc_amass)

    v.run()

def vis_diff_aitviewer(model_type, gt_full_poses, pred_full_poses, gt_betas=None, pred_betas=None, gt_trans=None, pred_trans=None):
    smpl_layer = SMPLLayer(model_type=model_type, gender="female", device=C.device)

    gt_smpl = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=gt_full_poses[:, 3:],
        poses_root=gt_full_poses[:, 0:3],
        betas=gt_betas,
        trans=gt_trans,
        z_up=True
    )
    offset = torch.tensor([[1.0,0.0,0.0]]).to(pred_full_poses.device)
    pred_smpl = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=pred_full_poses[:, 3:],
        poses_root=pred_full_poses[:, 0:3],
        betas=pred_betas,
        trans=pred_trans + offset,
        z_up=True
    )
    v = Viewer()
    v.scene.add(gt_smpl)
    v.scene.add(pred_smpl)
    v.run()



def vis_diff(pred_verts, gt_verts, faces):
    # visualize the difference between the output and the gt
    # 红色是gt，绿色是pred

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.3, 0.3, 0.3])
    pred_color = (0.2, 0.8, 0.2)
    gt_color = (0.8, 0.2, 0.2)
    def create_colored_mesh(verts, faces, color):
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        # 设置每个面的颜色
        color_rgba = np.array(list(color) + [1.0])  # 加alpha通道
        face_colors = np.tile((color_rgba * 255).astype(np.uint8), (faces.shape[0], 1))
        mesh.visual.face_colors = face_colors
        return pyrender.Mesh.from_trimesh(mesh, smooth=False)

    pred_mesh = create_colored_mesh(pred_verts, faces, pred_color)
    gt_mesh = create_colored_mesh(gt_verts, faces, gt_color)

    scene.add(pred_mesh, name='prediction')
    scene.add(gt_mesh, name='ground_truth')

    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

def create_coordinate_system(origin=[0, 0, 0], scale=0.1):
    """创建坐标系mesh - 使用圆柱体作为轴"""
    axes_meshes = []

    # X轴 (红色)
    x_cylinder = trimesh.creation.cylinder(
        radius=scale * 0.02,
        height=scale,
        transform=trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    )
    x_cylinder.apply_translation([origin[0] + scale / 2, origin[1], origin[2]])
    x_cylinder.visual.vertex_colors = [255, 0, 0, 255]  # 使用vertex_colors

    # Y轴 (绿色)
    y_cylinder = trimesh.creation.cylinder(
        radius=scale * 0.02,
        height=scale,
        transform=trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    )
    y_cylinder.apply_translation([origin[0], origin[1] + scale / 2, origin[2]])
    y_cylinder.visual.vertex_colors = [0, 255, 0, 255]  # 使用vertex_colors

    # Z轴 (蓝色)
    z_cylinder = trimesh.creation.cylinder(
        radius=scale * 0.02,
        height=scale
    )
    z_cylinder.apply_translation([origin[0], origin[1], origin[2] + scale / 2])
    z_cylinder.visual.vertex_colors = [0, 0, 255, 255]  # 使用vertex_colors

    # 合并三个轴
    coord_mesh = trimesh.util.concatenate([x_cylinder, y_cylinder, z_cylinder])

    return coord_mesh

def add_coordinate_systems_to_scene(scene, lcs, scale=0.05):
    '''
    :param scene: pyrender scene object
    :param lcs: (N, 4, 4) use SE(3) matrix to represent local coordinate system
    :param scale: a value to scale the arrow length
    :return:
    '''

    for i, item in enumerate(lcs):
        # 创建坐标系
        coord_mesh = create_coordinate_system([0, 0, 0], scale)

        # 创建变换矩阵
        transform = item

        # 添加到场景
        coord_node = pyrender.Node(
            mesh=pyrender.Mesh.from_trimesh(coord_mesh, smooth=False),
            matrix=transform
        )
        scene.add_node(coord_node)