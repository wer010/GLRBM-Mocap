import os
import pickle
import torch
import numpy as np
import argparse
import json
from glob import glob
import os.path as osp
from data import rigidbody_marker_id, moshpp_marker_id, BabelDataset, MetaBabelDataset, virtual_marker, MetaCollate
from torch.utils.data import DataLoader, random_split
from models import Moshpp, FrameModel, SequenceModel
from metric import MetricsEngine
from smpl import Smpl
from tqdm import tqdm
from utils import visualize, vis_diff
from datetime import datetime
from utils import visualize_aitviewer, vis_diff_aitviewer
import torch.optim as optim
from tensorboardX import SummaryWriter
import higher
from geo_utils import estimate_lcs_with_faces
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

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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

        scheduler.step()

        # evaluate the query set (support set)

        eval_result = test(
            test_dataset = test_dataset,
            model = model,
            smpl_model = smpl_model,
            metrics_engine = metrics_engine,
            device = device,
            vis=False,
            epochs_ft=1,
        )
        for key, value in eval_result["qry_set_overall"].items():
            writer.add_scalar(f"qry_set_overall/{key}", value, epoch)
        if eval_result["qry_set_overall"]["MPJPE [mm]"] < best_mpjpe:
            best_mpjpe = eval_result["qry_set_overall"]["MPJPE [mm]"]
            best_epoch = epoch

            print('*****************Best model saved*****************')
            torch.save(model.state_dict(), osp.join(save_dir, "model.pth"))
    with open(osp.join(save_dir, "best_epoch.txt"), "w") as f:
        f.write(f'The best epoch is {best_epoch}.')

def test(
    test_dataset,
    model,
    smpl_model,
    metrics_engine,
    model_path=None,
    device="cuda",
    vis=False,
    epochs_ft=1,
    batch_size=5,
    eval_supp_set=False,
):

    collate_fn = MetaCollate()
    testloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    if model_path is not None:
        model.load_state_dict(
            torch.load(osp.join(model_path, "model.pth"), map_location=device)
        )

    if eval_supp_set:
        eval_results = {
            "supp_set": [],
            "qry_set": []
        }
    else:
        eval_results = {
            "qry_set": []
        }
    for data in testloader:
        supp_set = {key: value[0] for key, value in data[0].items()}
        qry_set = {key: value[0] for key, value in data[1].items()}
        # print('Begin finetuning')
        B = supp_set["marker_info"].shape[0]
        L = supp_set["marker_info"].shape[1]
        ft_model = model.clone()
        optimizer = optim.Adam(ft_model.parameters(), lr=5e-4)

        ft_model.train()
        for e in range(epochs_ft):
            for ft_i in range(B // batch_size):
                optimizer.zero_grad()
                x = (
                    supp_set["marker_info"][batch_size * ft_i : (ft_i + 1) * batch_size]
                    .contiguous()
                    .view(batch_size, L, -1)
                )
                output = ft_model(x)
                gt = {
                    key: value[batch_size * ft_i : (ft_i + 1) * batch_size]
                    for key, value in supp_set.items()
                    if key != "task_name"
                }
                losses = loss_fn(output, gt, smpl_model)
                losses["total_loss"].backward()
                optimizer.step()

        ft_model.eval()
        if eval_supp_set:
            eval_data = {
                'supp_set': supp_set,
                'qry_set': qry_set,
                }
        else:
            eval_data = {
                'qry_set': qry_set,
            }
        # evaluate the support/query set
        for set_name, data in eval_data.items():
            B = data["marker_info"].shape[0]
            L = data["marker_info"].shape[1]
            output_list = []
            # 逐个sequence进行评估
            for val_i in range(B):
                x = data["marker_info"][val_i].contiguous().view(1, L, -1)
                output = ft_model(x)
                output["joints"] = smpl_model(
                    betas=output["betas"].reshape(-1),
                    body_pose=output["poses"].reshape(L, -1)[:, 3:],
                    global_orient=output["poses"].reshape(L, -1)[:, 0:3],
                    transl=output["trans"].reshape(L, -1),
                )["joints"].reshape(1, L, -1, 3)
                output_list.append(output)
                # 可视化当前sequence
                if vis:
                    vis_diff_aitviewer(
                        "smpl",
                        gt_full_poses=data["poses"][val_i],
                        gt_betas=data["betas"][val_i],
                        gt_trans=data["trans"][val_i],
                        pred_full_poses=output["poses"].squeeze(),
                        pred_betas=output["betas"].squeeze(),
                        pred_trans=output["trans"].squeeze(),
                    )
            # 合并所有sequence的输出，一次性计算评估指标
            outputs = {}
            for key in output_list[0].keys():
                outputs[key] = torch.concatenate([item[key] for item in output_list])
            metrics = metrics_engine.compute(outputs, data)
            print(
                metrics_engine.to_pretty_string(
                    metrics,
                    f"Task {supp_set['task_name']}-{model.model_name()}-{set_name}",
                )
            )
            eval_results[set_name].append(metrics)

    oa_results = {}
    ret = {}
    for key, item in eval_results.items():
        for k in item[0].keys():
            oa_results[k] = np.mean([i[k] for i in item])
        print(metrics_engine.to_pretty_string(oa_results, f"Overall {model.model_name()}-{key}"))
        ret[key + "_overall"] = oa_results

    return ret


def main(config):
    # 加载设备
    device = config.device if hasattr(config, "device") else "cuda"
    marker_type = config.marker_type if hasattr(config, "marker_type") else "moshpp"
    if marker_type == "moshpp":
        vid = [value for value in moshpp_marker_id.values()]
        config.data_path = osp.join(config.data_path, "moshpp")
        input_dim = 3
    elif marker_type == "rbm":
        vid = [value for value in rigidbody_marker_id.values()]
        config.data_path = osp.join(config.data_path, "rbm")
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
            only_pose=config.only_pose
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
            only_pose=config.only_pose
        ).to(device)
    else:
        raise ValueError(f"未知的base_model: {config.base_model}")

    metrics_engine = MetricsEngine()
    if config.only_pose:
        train_fp = osp.join(config.data_path, "meta_train_data_with_normalize_betas_marker.pkl")
        test_fp = osp.join(config.data_path, "meta_val_data_with_normalize_betas_marker.pkl")
    else:
        train_fp = osp.join(config.data_path, "metatrain.pkl")
        test_fp = osp.join(config.data_path, "metatest.pkl")


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
        train_dataset = BabelDataset(train_fp, use_rela_x=config.use_rela_x, marker_type=config.marker_type, device=device)
        test_dataset = MetaBabelDataset(test_fp, use_rela_x=config.use_rela_x, marker_type=config.marker_type, device=device)
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
        test_dataset = MetaBabelDataset(test_fp, use_rela_x=config.use_rela_x, marker_type=config.marker_type, device=device)
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
        epochs_ft=config.epochs_ft,
        eval_supp_set=True
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
        default="data/mosr/",
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
        "--only_pose", type=bool, default=False, help="Only predict pose."
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
        "--batch_size", type=int, default=5, help="Batch size for pretraining."
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
        default="moshpp",
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
