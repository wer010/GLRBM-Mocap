import os
import pickle
import torch
import numpy as np
import argparse
import json
import os.path as osp
from data import rigidbody_marker_id, moshpp_marker_id, virtual_marker, AmassLmdbDataset, train_collate_fn, rbm_a_config, rbm_b_config, rbm_c_config, rbm_d_config
from torch.utils.data import DataLoader, random_split
from geo_utils import rigid_landmark_transform_torch
from models import Moshpp, FrameModel, SequenceModel
from metric import MetricsEngine
from smpl import Smpl
from tqdm import tqdm
from datetime import datetime
from utils import visualize_aitviewer, vis_diff_aitviewer, vis_rbm_aitviewer
import torch.optim as optim
from tensorboardX import SummaryWriter
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import pandas as pd
from data import rigidbody_marker_id, virtual_marker, rela_x_fn
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle,axis_angle_to_quaternion, quaternion_invert, quaternion_multiply, quaternion_to_axis_angle


rbm_map = {
    "chest": "Global Angle body-breast:body-breast",
    "left_shin": "Global Angle body-calf-left:body-calf-left",
    "right_shin": "Global Angle body-calf-right:body-calf-right",
    "left_foot": "Global Angle body-foot-left:body-foot-left",
    "right_foot": "Global Angle body-foot-right:body-foot-right",
    "left_forearm": "Global Angle body-forearm-left:body-forearm-left",
    "right_forearm": "Global Angle body-forearm-right:body-forearm-right",
    "left_hand": "Global Angle body-hand-left:body-hand-left",
    "right_hand": "Global Angle body-hand-right:body-hand-right",
    "head": "Global Angle body-head:body-head",
    "left_thigh": "Global Angle body-thigh-left:body-thigh-left",
    "right_thigh": "Global Angle body-thigh-right:body-thigh-right",
    "left_arm": "Global Angle body-uparm-left:body-uparm-left",
    "right_arm": "Global Angle body-uparm-right:body-uparm-right",
}


import torch
betas_yjl = torch.tensor(
    [
        -0.55045104,
        -0.52951753,
        -0.41248605,
        0.31506306,
        -0.26099595,
        -0.48031747,
        -1.3245354,
        -0.14434478,
        0.68180835,
        0.00956408,
    ]
)
def geodesic_dis_quaternion(q1, q2):
    rel_rot = quaternion_multiply(q1, quaternion_invert(q2))  # (L-1, N, 4)
    rel_angle = 2 * torch.acos(torch.clamp(rel_rot[..., 0], -1, 1))  # (L-1, N)
    rel_angle_deg = torch.rad2deg(rel_angle)
    return rel_angle_deg


def quaternion_slerp(q1, q2, t, eps=1e-8):
    """
    批量四元数球面线性插值 (SLERP)

    Args:
        q1: (..., 4) 起始四元数
        q2: (..., 4) 目标四元数
        t: float or tensor ∈ [0,1] 插值系数
        eps: 防止数值误差

    Returns:
        q_interp: (..., 4) 插值后的四元数
    """
    # 单位化
    q1 = q1 / (q1.norm(dim=-1, keepdim=True) + eps)
    q2 = q2 / (q2.norm(dim=-1, keepdim=True) + eps)

    # 点积
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)

    # 如果点积为负，反转其中一个四元数，取最短路径
    mask = dot < 0.0
    q2 = torch.where(mask, -q2, q2)
    dot = torch.abs(dot)

    # 限制 dot 在 [-1, 1] 避免 acos NaN
    dot = torch.clamp(dot, -1.0, 1.0)

    # 计算角度
    theta_0 = torch.acos(dot)  # 原始角度
    sin_theta_0 = torch.sin(theta_0)

    # 若角度很小，退化为线性插值
    small_mask = sin_theta_0 < eps
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=q1.device, dtype=q1.dtype)
    t = t.view([-1] + [1] * (q1.ndim - 1)) if t.ndim == 0 else t

    t = t.expand_as(dot)

    # 正常SLERP公式
    s0 = torch.sin((1.0 - t) * theta_0) / (sin_theta_0 + eps)
    s1 = torch.sin(t * theta_0) / (sin_theta_0 + eps)

    q_interp = s0 * q1 + s1 * q2

    # 对小角度情形做LERP近似
    q_interp = torch.where(
        small_mask,
        (1.0 - t) * q1 + t * q2,
        q_interp
    )

    return q_interp / (q_interp.norm(dim=-1, keepdim=True) + eps)

def repair_axis_angle_jumps(axis_angles, angle_thresh_deg=15):
    """
    检测并修复姿态序列中的突变（跳变）

    Args:
        axis_angles: (L, N, 3)
        angle_thresh_deg: 检测突变的角度阈值（度）

    Returns:
        axis_angles_repaired: (L, N, 3)
    """
    L, N, _ = axis_angles.shape
    quats = axis_angle_to_quaternion(axis_angles)

    # 保持符号连续（避免四元数翻转）
    for t in range(1, L):
        dot = torch.sum(quats[t] * quats[t - 1], dim=-1, keepdim=True)
        flip_mask = dot < 0
        # 使用 squeeze(-1) 避免意外移除维度
        quats[t, flip_mask.squeeze(-1)] *= -1

    rel_angle_deg = geodesic_dis_quaternion(quats[1:], quats[:-1])

    # 找突变帧（角度过大）
    jump_mask = rel_angle_deg > angle_thresh_deg  # (L-1, N)

    repaired = quats.clone() 

    for n in range(N):
        # 找出所有突变位置（jump_mask中True的位置表示第idx帧到idx+1帧之间有突变）
        jump_indices = torch.nonzero(jump_mask[:, n], as_tuple=False).flatten()+1
        if jump_indices.numel() == 0:
            continue

        # 合并相邻或接近的突变位置
        # 如果两个突变位置之间间隔较小（<=2），说明它们可能属于同一个突变段
        # 例如：[4, 6] 如果间隔只有2，应该合并为 [4, 6]，表示突变段 [5, 6]
        jump_groups = []
        cur = []
        for idx in jump_indices.tolist():
            if not cur or idx <= cur[-1] + 5:
                # 间隔 <= 2，可能是同一个突变段，继续当前组
                cur.append(idx)
            else:
                # 间隔 > 2，开始新组
                jump_groups.append(cur)
                cur = [idx]
        if cur:
            jump_groups.append(cur)
        
        # 对每个突变段进行处理
        for jump_group in jump_groups:
            if len(jump_group) == 0:
                continue
            

            first_jump = jump_group[0]  # 第一个突变位置
            last_jump = jump_group[-1]   # 最后一个突变位置
            
            
            # 确定修复范围：使用突变段两侧的有效帧作为参考
            left = max(first_jump - 1, 0)  # 突变段左侧的有效参考帧（突变前）
            right = min(last_jump+1, L - 1)  # 突变段右侧的有效参考帧（突变后）
            
            if right <= left:
                continue
            
            # SLERP 插值修复：在 left 和 right 之间进行球面线性插值
            # 修复范围包括整个突变段（start 到 end 的所有帧）
            q1 = repaired[left, n]  # 左侧参考四元数
            q2 = repaired[right, n]  # 右侧参考四元数
            

            for t in range(left + 1, right):
                # 计算插值系数：t 在 [left, right] 区间中的归一化位置
                alpha = (t - left) / (right - left) if right > left else 0.0
                repaired[t, n] = quaternion_slerp(q1.unsqueeze(0), q2.unsqueeze(0), alpha)[0]

    return quaternion_to_axis_angle(repaired)

def load_data(data_path):
    df = pd.read_csv(data_path,sep = ',',skiprows=2, header=None)
    start_col_id = 2
    num_rbms = len(df.columns)//6
    data = {}
    for i in range(num_rbms):
        key = df.iloc[0, start_col_id+i*6]
        value = {
            "ori": df.iloc[3:, start_col_id+i*6:start_col_id+i*6+3].astype(float).to_numpy(),
            "pos": df.iloc[3:, start_col_id+i*6+3:start_col_id+i*6+6].astype(float).to_numpy(),
        }
        data[key] = value
    data_list = []
    # 根据部位名称对数据进行重新排序
    for key, value in rigidbody_marker_id.items():
        data_list.append(data[rbm_map[key]])
    ret = {
        "ori": np.stack([item["ori"] for item in data_list], axis=1),
        "pos": np.stack([item["pos"]/1000.0 for item in data_list], axis=1),
    }
    # 将无效值替换为0
    mask = np.ones(ret["pos"].shape[1], dtype=bool)
    mask[-3:] = False
    mask[2:5] = False
    for key, value in ret.items():
        np_value = np.nan_to_num(value)
        ret[key] = torch.from_numpy(np_value[::3,:,:]).float()
        # ret[key] = ret[key][:, mask, :]
    # 可视化数据

    # vis_rbm_aitviewer([{'pos': ret["pos"][:, i:i+1, :], 'ori': ret["ori"][:, i:i+1, :]} for i in range(ret["pos"].shape[1])],z_up=True)

    # ret["ori"] = repair_axis_angle_jumps(ret["ori"])
    
    return ret


def rbm_calibration(real_markers):
    # two step calibration 1. use procrustes analysis to align the ori and pos 2. calculate the se3 matrix between the real data and the virtual data
    ind = 50
    real_markers_pos = real_markers["pos"]
    real_markers_ori = real_markers["ori"]
    real_markers_ori = axis_angle_to_matrix(real_markers_ori)

    template_markers_pos = real_markers_pos[ind,...]
    template_markers_ori = real_markers_ori[ind,...]


    vid = [value for value in rigidbody_marker_id.values()]
    n_marker = len(vid)
    pos_offset = torch.tensor([0.0095, 0, 0, 1]).expand([n_marker, -1])
    ori_offset = torch.eye(3).expand([n_marker, -1, -1])
    vid_tensor = torch.tensor(vid)
    pose = torch.zeros(1, 24*3)
    pose[:, 0:3] = torch.tensor([1.57079633, -0.        , -0.])
    trans = torch.tensor([[0.0000,  0.0000,  0.88]])
    virtual_marker_pos, virtual_marker_ori, virtual_v_posed, virtual_joints = virtual_marker(betas_yjl,
                                                                pose,
                                                                trans,
                                                                vid_tensor,
                                                                pos_offset,
                                                                ori_offset,
                                                                visualize_flag=False)

    # visualize_aitviewer("smpl", pose, betas_yjl, trans, rbs={'pos': virtual_marker_pos.squeeze()[None,...], 'ori': virtual_marker_ori.squeeze()[None,...]})
    virtual_marker_pos = virtual_marker_pos.squeeze()
    virtual_marker_ori = virtual_marker_ori.squeeze()

    rel_rot, rel_trans = rigid_landmark_transform_torch(virtual_marker_pos.T, template_markers_pos.T)

    aligned_virtual_markers_pos = rel_rot @ virtual_marker_pos[...,None] + rel_trans
    aligned_virtual_markers_ori = rel_rot @ virtual_marker_ori
    offset_ori = torch.matmul(aligned_virtual_markers_ori.permute(0,2,1), template_markers_ori.squeeze()) 
    offset_pos = torch.matmul(aligned_virtual_markers_ori.permute(0,2,1), (template_markers_pos.squeeze() - aligned_virtual_markers_pos.squeeze())[...,None]) 

    cali_real_ori = torch.matmul(real_markers_ori, offset_ori.permute(0,2,1))
    cali_real_pos = real_markers_pos - torch.matmul(cali_real_ori, offset_pos).squeeze()

    template_markers = {'pos': template_markers_pos[None,...], 'ori': template_markers_ori[None,...]}
    virtual_markers = {'pos': virtual_marker_pos.squeeze()[None,...], 'ori': virtual_marker_ori.squeeze()[None,...]}
    aligned_virtual_markers = {'pos': aligned_virtual_markers_pos.squeeze()[None,...], 'ori': aligned_virtual_markers_ori.squeeze()[None,...]}
    all_cali_real_markers = {'pos': cali_real_pos, 'ori': cali_real_ori}

    cali_real_markers = {'pos': cali_real_pos[ind,...][None,...], 'ori': cali_real_ori[ind,...][None,...]}
    # vis_rbm_aitviewer([all_cali_real_markers], z_up=True)
    vis_rbm_aitviewer([template_markers, virtual_markers, aligned_virtual_markers, cali_real_markers], z_up=True)

    # transform the virtual markers to the real markers
    return cali_real_pos, matrix_to_axis_angle(cali_real_ori)

def test(
    test_data,
    model,
    smpl_model,
    model_path=None,
    device="cuda",
):

    if model_path is not None:
        model.load_state_dict(
            torch.load(osp.join(model_path, "model.pth"), map_location=device)
        )

    model.eval()

    L = test_data["pos"].shape[0]
    x = torch.cat([test_data["pos"], test_data["ori"]], dim=-1).to(device)
    x = rela_x_fn(x, marker_type='rbm').contiguous().view(1, L, -1)


    # 逐个sequence进行评估
    if L>120:
        n_slices = int(np.ceil(L/120))
        output_list = []
        for i in range(n_slices):
            if i == 0:
                is_new_sequence = True
            else:
                is_new_sequence = False
            chunk_x = x[:, 120*i:120*(i+1),:].contiguous()
            seq_len = chunk_x.shape[1]
            output = model(chunk_x, is_new_sequence=is_new_sequence)


            output_list.append(output)
        output = {}
        for key in output_list[0].keys():
            if key == 'betas':
                output[key] = torch.mean(torch.stack([item[key].squeeze() for item in output_list]), dim=0, keepdim=True)
            else:
                output[key] = torch.concatenate([item[key] for item in output_list], dim=1)

    else:
        output = model(x, is_new_sequence=True)



    
    # 可视化当前sequence

    visualize_aitviewer(
        "smpl",
        output["poses"][0],
        betas_yjl,
        output["trans"][0],
        rbs={'pos': test_data["pos"], 'ori': test_data["ori"]},
    )
    # 合并所有sequence的输出，一次性计算评估指标


def main(config):
    # 加载设备
    device = config.device if hasattr(config, "device") else "cuda"
    # 根据测试文件夹下的config.json自动加载测试模式下的配置
    json_path = osp.join(config.model_path, "config.json")
    with open(json_path, "r") as f:
        saved_config = json.load(f)
    config.marker_type = saved_config["marker_type"]
    config.base_model = saved_config["base_model"]
    config.model_type = saved_config["model_type"]
    config.use_rela_x = saved_config["use_rela_x"]
    config.hidden_size = saved_config["hidden_size"]
    config.num_layers = saved_config["num_layers"]
    config.dropout = saved_config["dropout"]
    
    
    if config.marker_type == "moshpp":
        vid = np.array([value for value in moshpp_marker_id.values()], dtype=np.int32)
        input_dim = 3
    elif "rbm" in config.marker_type:
        vid = np.array([value for value in rigidbody_marker_id.values()], dtype=np.int32)
        input_dim = 6
        if config.marker_type == "rbm_a":
            vid = vid[rbm_a_config]
            input_dim = 6
        elif config.marker_type == "rbm_b":
            vid = vid[rbm_b_config]
        elif config.marker_type == "rbm_c":
            vid = vid[rbm_c_config]
        elif config.marker_type == "rbm_d":
            vid = vid[rbm_d_config]
    else:
        raise ValueError(f"未知的marker_type: {config.marker_type}")
    n_marker = len(vid)

    if config.use_rela_x:
        input_dim = input_dim*n_marker + 3
    else:
        input_dim = input_dim*n_marker

 
    # 根据配置选择模型
    if config.base_model == "sequence":
        model = SequenceModel(
            input_size=input_dim,
            betas_size=10,
            poses_size=24 * 3,
            trans_size=3,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            m_dropout=config.dropout if hasattr(config, "dropout") else 0.0,
            m_bidirectional=True,
            model_type=config.model_type,
        ).to(device)
    elif config.base_model == "frame":
        model = FrameModel(
            input_size=input_dim,
            betas_size=10,
            poses_size=24 * 3,
            trans_size=3,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            m_dropout=config.dropout if hasattr(config, "dropout") else 0.0,
        ).to(device)
    else:
        raise ValueError(f"未知的base_model: {config.base_model}")

    metrics_engine = MetricsEngine()

    test_data = load_data(config.data_path)
    cali_real_pos, cali_real_ori = rbm_calibration(test_data)

    test_data["pos"] = cali_real_pos
    test_data["ori"] = cali_real_ori

    smpl_model = Smpl(
            model_path=config.smpl_model_path,
            device=device,
        )
    

    assert config.model_path is not None, "model_path is required for test mode"
    test_dir = config.model_path


    test(
        test_data = test_data,
        model = model,
        smpl_model = smpl_model,
        model_path = test_dir,
        device = device,
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -----------------------
    # General experiment setup
    # -----------------------

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/real_data/1105/subject1-tpose-static2.csv",
        help="Path to data.",
    )
    parser.add_argument(
        "--smpl_model_path",
        type=str,
        default="data/models/smpl/SMPL_FEMALE.npz",
        help="Path to smpl model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='results/20251103-1908-sequence-train-rbm-1000epochs',
        help="Path to trained model.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use: cuda or cpu."
    )

    config = parser.parse_args()
    main(config)
