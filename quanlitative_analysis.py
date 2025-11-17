import argparse
import json
import os.path as osp
from datetime import datetime
import numpy as np
from data import rigidbody_marker_id, moshpp_marker_id, rbm_a_config, rbm_b_config, rbm_c_config, rbm_d_config, rbm_e_config, rbm_f_config
from models import SequenceModel, FrameModel, ContinuitySequenceModel
from metric import MetricsEngine
from smpl import Smpl
from ptflops import get_model_complexity_info


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
        elif config.marker_type == "rbm_e":
            vid = vid[rbm_e_config]
        elif config.marker_type == "rbm_f":
            vid = vid[rbm_f_config]
    else:
        raise ValueError(f"未知的marker_type: {config.marker_type}")
    n_marker = len(vid)

    if config.use_rela_x:
        input_dim = input_dim*n_marker + 3
    else:
        input_dim = input_dim*n_marker

    # 保存目录
    save_dir = osp.join("./results", f'{datetime.now().strftime("%Y%m%d-%H%M")}-{config.base_model}-{config.train_mode}-{config.marker_type}-{config.epochs}epochs')

    # 根据配置选择模型
    use_continuity_loss = False
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
    elif config.base_model == "continuity":
        model = ContinuitySequenceModel(
            input_size=input_dim,
            betas_size=10,
            poses_size=24 * 3,
            trans_size=3,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            m_dropout=config.dropout if hasattr(config, "dropout") else 0.0,
            m_bidirectional=True,
            model_type=config.model_type,
            rotation_mode = 'ortho6d'
        ).to(device)
        use_continuity_loss = True
    else:
        raise ValueError(f"未知的base_model: {config.base_model}")


    macs, params = get_model_complexity_info(model, ( 120, input_dim), as_strings=True, print_per_layer_stat=True)
    print(
        f"Model: {model.model_name()}. Marker type: {config.marker_type}. Base model: {config.base_model}. Model type: {config.model_type}. Use rela x: {config.use_rela_x}. Hidden size: {config.hidden_size}. Num layers: {config.num_layers}. Dropout: {config.dropout}."
    )
    print("MACs: ", macs)
    print("Params: ", params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        default='results/20251019-1041-sequence-train-rbm-1000epochs',
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
        choices=["frame", "sequence","continuity"],
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
        "--use_rela_x", action="store_true", help="Use relative x."
    )
    parser.add_argument(
        "--use_geodesic_loss", action="store_true", help="Use geodesic loss.",
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
    "--lr_decay",
    action="store_true",
    help="Use learning rate decay."    
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
        choices=["moshpp", "rbm", "rbm_a", "rbm_b", "rbm_c", "rbm_d", "rbm_e", "rbm_f"],
        help="Marker type.",
    )

    args = parser.parse_args()
    main(args)
