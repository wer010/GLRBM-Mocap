import os
import pickle
import torch
import numpy as np
import argparse
import json
import os.path as osp
from data import rigidbody_marker_id, moshpp_marker_id, virtual_marker, AmassLmdbDataset, train_collate_fn
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


def geodesic_loss(aa1, aa2):
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

    distance = 4*(1 - cos_dist**2)

    return torch.mean(distance)


def loss_fn(output, gt, smpl_model=None, do_fk=True):
    B = output["poses"].shape[0]
    L = output["poses"].shape[1]
    device = output["poses"].device

    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    pose_loss = mse_loss(output["poses"], gt["poses"])
    shape_loss = l1_loss(output["betas"].view(B, -1), gt["betas"])
    tran_loss = mse_loss(output["trans"], gt["trans"])
    angle_loss = geodesic_loss(output["poses"].reshape(B, L, -1, 3), gt["poses"].reshape(B, L, -1, 3))
    if do_fk:
        joints_hat = smpl_model(
            betas=output["betas"].expand(-1, L, -1).reshape(B * L, -1),
            body_pose=output["poses"].reshape(B * L, -1)[:, 3:],
            global_orient=output["poses"].reshape(B * L, -1)[:, 0:3],
            transl=output["trans"].reshape(B * L, -1),
        )["joints"].reshape(B, L, -1, 3)
        fk_loss = mse_loss(joints_hat, gt["joints"])
    else:
        fk_loss = torch.zeros(1, device=device)
    total_loss = angle_loss + shape_loss + tran_loss + 0.1 * fk_loss

    losses = {
        "pose": angle_loss,
        "shape": shape_loss,
        "tran": tran_loss,
        "fk": fk_loss,
        "total_loss": total_loss,
    }
    return losses


def train(
    train_dataset,
    test_dataset,
    model,
    smpl_model,
    save_dir,
    metrics_engine,
    batch_size=5,
    device="cuda",
    lr=5e-4,
    epochs=400,
):
    writer = SummaryWriter(os.path.join(save_dir, "logs"))
    best_mpjpe = torch.inf
    # 普通训练

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)
    train_optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(train_optimizer, step_size=epochs//10, gamma=0.8)
    global_step = 0
    print("Begin training.")
    for epoch in tqdm(range(epochs)):
        model.train()

        for data in trainloader:
            B = data["marker_info"].shape[0]
            L = data["marker_info"].shape[1]
            x = data["marker_info"].contiguous().view(B, L, -1)

            train_optimizer.zero_grad()
            global_step += 1

            output = model(x)

            losses = loss_fn(output, data, smpl_model)
            if writer is not None:
                mode_prefix = "train"
                for k in losses:
                    prefix = "{}/{}".format(k, mode_prefix)
                    writer.add_scalar(prefix, losses[k].cpu().item(), global_step)

            writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
            losses["total_loss"].backward()
            train_optimizer.step()

        # scheduler.step()

        # evaluate the query set (support set)
        eval_result = test(
                        test_dataset = test_dataset,
                        model = model,
                        smpl_model = smpl_model,
                        metrics_engine = metrics_engine,
                        model_path = None,
                        device = device,
                        vis=False,
                        val=True
                    )
        for key, value in eval_result.items():
            writer.add_scalar(f"Test/{key}", value, epoch)
        if eval_result["MPJPE [mm]"] < best_mpjpe:
            best_mpjpe = eval_result["MPJPE [mm]"]
            best_epoch = epoch
            best_result = eval_result

            print('*****************Best model saved*****************')
            torch.save(model.state_dict(), osp.join(save_dir, "model.pth"))
    with open(osp.join(save_dir, "best_epoch.txt"), "w") as f:
        f.write(f'The best epoch is {best_epoch}.\n')
        f.write(json.dumps(best_result, indent=4, ensure_ascii=False))

def test(
    test_dataset,
    model,
    smpl_model,
    metrics_engine,
    model_path=None,
    device="cuda",
    vis=False,
    val=False
):

    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    if model_path is not None:
        model.load_state_dict(
            torch.load(osp.join(model_path, "model.pth"), map_location=device)
        )

    model.eval()
    eval_results = []
    for data_idx, data in enumerate(testloader):
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
                    output[key] = torch.mean(torch.stack([item[key].squeeze() for item in output_list]), dim=0)
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
    marker_type = config.marker_type if hasattr(config, "marker_type") else "moshpp"
    if marker_type == "moshpp":
        vid = [value for value in moshpp_marker_id.values()]
        input_dim = 3
    elif marker_type == "rbm":
        vid = [value for value in rigidbody_marker_id.values()]
        input_dim = 6
    else:
        raise ValueError(f"未知的marker_type: {marker_type}")
    n_marker = len(vid)

    if config.use_rela_x:
        input_dim = input_dim*n_marker + 3
    else:
        input_dim = input_dim*n_marker

    # 保存目录
    save_dir = osp.join("./results", f'{datetime.now().strftime("%Y%m%d-%H%M")}-{config.base_model}-{config.train_mode}-{config.marker_type}-{config.epochs}epochs')

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

    train_fp = osp.join(config.data_path, "CMU_lmdb_"+config.marker_type)
    test_fp = osp.join(config.data_path, "BMLrub_lmdb_"+config.marker_type)


    smpl_model = Smpl(
            model_path=config.smpl_model_path,
            device=device,
        )
    
    # 根据训练模式选择训练方式
    if config.train_mode == "train":
        # 保存目录
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
            with open(osp.join(save_dir, "config.json"), "w") as f:
                json.dump(config.__dict__, f)
        train_dataset = AmassLmdbDataset(train_fp, use_rela_x=config.use_rela_x, marker_type=config.marker_type, device=device)
        test_dataset = AmassLmdbDataset(test_fp, use_rela_x=config.use_rela_x, marker_type=config.marker_type, device=device)
        train(
            train_dataset,
            test_dataset,
            model,
            smpl_model,
            save_dir,
            metrics_engine,
            batch_size=config.batch_size,
            device=device,
            lr=config.lr,
            epochs=config.epochs,
        )
        test_dir = save_dir
    elif config.train_mode == "test":
        # 如果有指定测试目录则用，否则用当前save_dir
        assert config.model_path is not None, "model_path is required for test mode"
        test_dataset = AmassLmdbDataset(test_fp, use_rela_x=config.use_rela_x, device=device)
        test_dir = config.model_path
    else:
        raise ValueError(f"未知的train_mode: {config.train_mode}")

    # 测试模型
    test(
        test_dataset = test_dataset,
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
        default="data/",
        help="Path to dataset.",
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
        default=None,
        help="Path to trained model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed.")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use: cuda or cpu."
    )

    # -----------------------
    # Training mode
    # -----------------------
    parser.add_argument(
        "--train_mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Training mode: train (standard) or test (meta-learning).",
    )

    # -----------------------
    # Model config
    # -----------------------
    parser.add_argument(
        "--base_model",
        type=str,
        default="sequence",
        choices=["frame", "sequence"],
        help="Backbone model type.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="lstm",
        choices=["rnn", "lstm", "gru", "cnn", "transformer"],
        help="Model type.",
    )

    parser.add_argument(
        "--use_rela_x", type=bool, default=True, help="Use relative x."
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Hidden size for RNN/MLP layers."
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of RNN layers."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout probability."
    )

    # -----------------------
    # Pretrain config
    # -----------------------
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs for pretraining."
    )
    parser.add_argument(
        "--batch_size", type=int, default=15, help="Batch size for pretraining."
    )

    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="step",
        choices=["step", "cosine", "plateau", "none"],
        help="Learning rate scheduler type.",
    )

    
    # Input data.
    parser.add_argument(
        "--marker_type",
        type=str,
        default="rbm",
        choices=["moshpp", "rbm"],
        help="Marker type.",
    )

    parser.add_argument(
        "--use_real_offsets",
        action="store_true",
        help="Sampling is informed by real offset distribution.",
    )
    parser.add_argument(
        "--offset_noise_level",
        type=int,
        default=0,
        help="How much noise to add to real offsets.",
    )

    # Data augmentation.
    parser.add_argument(
        "--noise_num_markers",
        type=int,
        default=1,
        help="How many markers are affected by the noise.",
    )
    parser.add_argument(
        "--spherical_noise_strength",
        type=float,
        default=0.0,
        help="Magnitude of noise in %.",
    )
    parser.add_argument(
        "--spherical_noise_length",
        type=float,
        default=0.0,
        help="Temporal length of noise in %.",
    )
    parser.add_argument(
        "--suppression_noise_length",
        type=float,
        default=0.0,
        help="Marker suppression length.",
    )
    parser.add_argument(
        "--suppression_noise_value",
        type=float,
        default=0.0,
        help="Marker suppression value.",
    )

    config = parser.parse_args()
    main(config)
