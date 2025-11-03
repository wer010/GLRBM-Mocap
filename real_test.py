import os
import pickle
import torch
import numpy as np
import argparse
import json
import os.path as osp
from data import rigidbody_marker_id, moshpp_marker_id, virtual_marker, AmassLmdbDataset, train_collate_fn, rbm_a_config, rbm_b_config, rbm_c_config, rbm_d_config
from torch.utils.data import DataLoader, random_split
from models import Moshpp, FrameModel, SequenceModel
from metric import MetricsEngine
from smpl import Smpl
from tqdm import tqdm
from datetime import datetime
from utils import visualize_aitviewer, vis_diff_aitviewer
import torch.optim as optim
from tensorboardX import SummaryWriter
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import pandas as pd


# TODO: 1、去除无效值；2、根据部位名称对数据进行排序；3、进行t-pose下的初始alignment
def load_data(data_path):
    df = pd.read_csv(data_path,sep = '\t')
    start_col_id = 2
    num_rbms = len(df.columns)//6
    data = {}
    for i in range(num_rbms):
        key = df.iloc[1, start_col_id+i*6]
        value = {
            "ori": df.iloc[4:, start_col_id+i*6:start_col_id+i*6+3].astype(float).to_numpy(),
            "pos": df.iloc[4:, start_col_id+i*6+3:start_col_id+i*6+6].astype(float).to_numpy(),
        }
        data[key] = value
    return data

def test(
    test_data,
    model,
    smpl_model,
    metrics_engine,
    model_path=None,
    device="cuda",
    vis=False,
    val=False
):

    if model_path is not None:
        model.load_state_dict(
            torch.load(osp.join(model_path, "model.pth"), map_location=device)
        )

    model.eval()
    eval_results = []
    for data_idx, data in enumerate(tqdm(test_data)):
        B = data["marker_info"].shape[0]
        L = data["marker_info"].shape[1]
        x = data["marker_info"].contiguous().view(B, L, -1)

        if val and data_idx>=50:
            break

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

                output["joints"] = smpl_model(
                betas=output["betas"].reshape(-1),
                body_pose=output["poses"].reshape(seq_len, -1)[:, 3:],
                global_orient=output["poses"].reshape(seq_len, -1)[:, 0:3],
                transl=output["trans"].reshape(seq_len, -1),
                )["joints"].reshape(1, seq_len, -1, 3)

                output_list.append(output)
            output = {}
            for key in output_list[0].keys():
                if key == 'betas':
                    output[key] = torch.mean(torch.stack([item[key].squeeze() for item in output_list]), dim=0, keepdim=True)
                else:
                    output[key] = torch.concatenate([item[key] for item in output_list], dim=1)
 
        else:
            output = model(x, is_new_sequence=True)

            output["joints"] = smpl_model(
            betas=output["betas"].reshape(-1),
            body_pose=output["poses"].reshape(L, -1)[:, 3:],
            global_orient=output["poses"].reshape(L, -1)[:, 0:3],
            transl=output["trans"].reshape(L, -1),
            )["joints"].reshape(1, L, -1, 3)

        
        # 可视化当前sequence
        if vis:
            vis_diff_aitviewer(
                "smpl",
                gt_full_poses=data["poses"][0],
                gt_betas=data["betas"][0],
                gt_trans=data["trans"][0],
                pred_full_poses=output["poses"][0],
                pred_betas=output["betas"][0],
                pred_trans=output["trans"][0],
            )
        # 合并所有sequence的输出，一次性计算评估指标
        
        metrics = metrics_engine.compute(output, data)
        eval_results.append(metrics)

    oa_results = {}
    for key in eval_results[0].keys():
        oa_results[key] = np.mean([item[key] for item in eval_results])
    print(metrics_engine.to_pretty_string(oa_results, f"Overall {model.model_name()}"))

    return oa_results


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
        metrics_engine = metrics_engine,
        model_path = test_dir,
        device = device,
        vis=False,
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
        default="data/real_data/subject-minibest-1.csv",
        help="Path to data.",
    )
    parser.add_argument(
        "--smpl_model_path",
        type=str,
        default="data/models/smpl/SMPL_NEUTRAL.npz",
        help="Path to smpl model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='results/20251014-1903-sequence-train-rbm-1000epochs',
        help="Path to trained model.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use: cuda or cpu."
    )

    config = parser.parse_args()
    main(config)
