import numpy as np
import torch
import copy
import os.path as osp
from loguru import logger
from omegaconf import OmegaConf
from smpl import Smpl
from geo_utils import estimate_lcs_with_curv, estimate_lcs_with_faces, invert_se3, rigid_landmark_transform_batch
from pytorch3d.ops.knn import  knn_points
import torch.nn.functional as F
from utils import visualize
import math

class StageOneFitter(torch.nn.Module):
    def __init__(self,
                 num_frames = 12,
                 num_betas=10,
                 device='cuda',
                 num_epochs = 4000):
        super().__init__()
        self.device = device
        self.num_betas = num_betas
        self.num_frames = num_frames
        self.model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz', device=device)
        self.num_epochs = num_epochs

    def fit(self,
            train_data,
            init_smpl_id,
            num_epochs=None):
        assert train_data.shape[0] == self.num_frames
        print(f"Starting Stage i training for {num_epochs} epochs.")
        if num_epochs is None:
            num_epochs = self.num_epochs

        n_marker = len(init_smpl_id)
        pos_offset = torch.tensor([0.0095, 0, 0, 1]).expand([n_marker, -1]).to(self.device)
        vid_tensor = torch.tensor(init_smpl_id)[None,:].to(self.device)
        lcs = estimate_lcs_with_faces(vid=vid_tensor,
                                      fid=self.model.vertex_faces[vid_tensor],
                                      vertices=self.model.v_template[None, ...],
                                      faces=self.model.faces_tensor)

        init_marker_pose = torch.matmul(lcs, pos_offset[None, ..., None])[:, :, 0:3, 0]
        self.marker_pose = torch.nn.Parameter(init_marker_pose)

        # visualize(self.model.vertices, self.model.faces,[init_marker_pose.cpu().detach().numpy(), data[0].cpu().detach().numpy()])


        # 1. find the nearest vertices to build up local coordinate system (LCS)
        self.betas = torch.nn.Parameter(torch.zeros(self.num_betas, device=self.device))
        self.body_pose = torch.nn.Parameter(torch.zeros([self.num_frames,23*3], device=self.device))

        global_orient, transl = rigid_landmark_transform_batch(init_marker_pose.expand(self.num_frames,-1,-1), train_data)
        self.global_orient = torch.nn.Parameter(global_orient)
        self.transl = torch.nn.Parameter(transl)

        # 2. given initial marker poses under the LCS
        # marker_pose_l = torch.tensor([[0.0095,0,0,1]]).repeat(len(id),1)

        # 2. obtain the marker poses under the global coordinate system (GCS)
        # init_marker_pose = torch.matmul(lcs, marker_pose_l[..., None])

        wt_data = (75 / 1) * (46 / 53)
        wt_pose = 3.0
        wt_betas = 10.0
        wt_dis = 10000.0
        print(self.parameters())
        opt = torch.optim.Adam(self.parameters())

        for i in range(num_epochs):
            opt.zero_grad()
            # dist, idx, _ = knn_points(self.marker_pose, self.model.v_template[None, ...], K=8, return_nn=False)
            # lcs = estimate_lcs_with_curv(idx, self.model.v_template, K=8)

            dist, idx, _ = knn_points(self.marker_pose, self.model.v_template[None, ...], K=1, return_nn=False)
            ids = idx[:, :, 0]
            lcs = estimate_lcs_with_faces(vid = ids,
                                          fid=self.model.vertex_faces[ids],
                                          vertices=self.model.v_template[None, ...],
                                          faces=self.model.faces_tensor)
            lcs_inv = invert_se3(lcs)

            marker_pose_l = torch.matmul(lcs_inv, F.pad(self.marker_pose, (0, 1), value=1)[..., None])

            # FK process: calculate the body surface vertices
            v_posed = self.model(betas=self.betas,
                                 body_pose=self.body_pose,
                                 global_orient=self.global_orient,
                                 transl=self.transl)['vertices']
            # lcs_posed = estimate_lcs_with_curv(idx, v_posed, K=8)
            batch_id =ids.expand(12,-1)
            lcs_posed = estimate_lcs_with_faces(vid = batch_id,
                                                fid=self.model.vertex_faces[batch_id],
                                                vertices=v_posed,
                                                faces=self.model.faces_tensor)
            # calculate the marker position
            marker_pose_g = torch.matmul(lcs_posed, marker_pose_l)
            # if i % 500 == 0:
            #     visualize(v_posed[0].cpu().detach().numpy(), self.model.faces,
            #               [marker_pose_g[0, :, 0:3, 0].cpu().detach().numpy(), data[0].cpu().detach().numpy()])

            # calculate the loss between prediction and measurement
            loss_data =  torch.sum(wt_data *(marker_pose_g[:,:, 0:3, 0] - train_data) ** 2)
            loss_beta = wt_betas*torch.sum(self.betas ** 2)
            loss_dis = torch.sum(wt_dis*(marker_pose_l[:,:,0,0]- 0.0095)**2)
            loss = torch.mean(loss_data + loss_beta + loss_dis)
            loss.backward()
            opt.step()
        logger.debug(f"Loss values {loss} = loss_data:{loss_data} + loss_beta{loss_beta} + loss_dis{loss_dis}")

        # print(self.body_pose.cpu().detach())
        # visualize(v_posed[0].cpu().detach().numpy(), self.model.faces,
        #                       [torch.matmul(lcs_posed, pos_offset[None, ..., None])[0, :, 0:3, 0].cpu().detach().numpy(), marker_pose_g[0, :, 0:3, 0].cpu().detach().numpy()])
        return self.marker_pose, self.betas

class StageTwoFitter(torch.nn.Module):
    def __init__(self,
                 num_frames=100,
                 num_betas=10,
                 device='cuda',
                 num_epochs = 1000):
        super().__init__()
        self.device = device
        self.num_betas = num_betas
        self.num_frames = num_frames
        self.model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz', device=device)
        self.num_epochs = num_epochs

    def fit(self, data, m, betas, num_epochs = None):
        print(f"Starting Stage ii training for {num_epochs} epochs.")
        if num_epochs is None:
            num_epochs = self.num_epochs

        self.register_buffer('mo', m.detach().clone())
        self.register_buffer('betas', betas.detach().clone())

        dist, idx, _ = knn_points(self.mo, self.model.v_template[None, ...], K=1, return_nn=False)
        ids = idx[:, :, 0]
        lcs = estimate_lcs_with_faces(vid=ids,
                                      fid=self.model.vertex_faces[ids],
                                      vertices=self.model.v_template[None, ...],
                                      faces=self.model.faces_tensor)
        lcs_inv = invert_se3(lcs)
        marker_pose_l = torch.matmul(lcs_inv, F.pad(self.mo, (0, 1), value=1)[..., None])

        n_slices = math.ceil(data.shape[0] / self.num_frames)
        poses_list = []
        transl_list = []
        joint_list = []
        for i in range(n_slices):
            wt_data = (75 / 1) * (46 / 53)
            wt_pose = 3.0
            wt_betas = 10.0
            wt_dis = 10000.0

            batch_data  = data[self.num_frames * i:self.num_frames * (i + 1)]
            batch_size = batch_data.shape[0]
            batch_id = ids.expand(batch_size, -1)

            body_pose = torch.nn.Parameter(torch.zeros([batch_size, 23 * 3], device=self.device))
            global_orient, transl = rigid_landmark_transform_batch(
                self.mo.expand(batch_size, -1, -1), batch_data)
            global_orient = torch.nn.Parameter(global_orient)
            transl = torch.nn.Parameter(transl)

            # print([body_pose, global_orient, transl])
            opt = torch.optim.Adam([body_pose, global_orient, transl])
            for j in range(num_epochs):
                opt.zero_grad()
                # FK process: calculate the body surface vertices
                v_posed = self.model(betas=self.betas,
                                     body_pose=body_pose,
                                     global_orient=global_orient,
                                     transl=transl)['vertices']
                # lcs_posed = estimate_lcs_with_curv(idx, v_posed, K=8)

                lcs_posed = estimate_lcs_with_faces(vid=batch_id,
                                                    fid=self.model.vertex_faces[batch_id],
                                                    vertices=v_posed,
                                                    faces=self.model.faces_tensor)
                # calculate the marker position
                marker_pose_g = torch.matmul(lcs_posed, marker_pose_l)

                # calculate the loss between prediction and measurement
                loss_data = torch.sum(wt_data * (marker_pose_g[:, :, 0:3, 0] - batch_data) ** 2)
                loss = torch.mean(loss_data)

                loss.backward()
                opt.step()
            logger.debug(
                f"Loss values {loss} = loss_data:{loss_data}")
            joint_list.append(self.model(betas=self.betas,
                                     body_pose=body_pose,
                                     global_orient=global_orient,
                                     transl=transl)['joints'])
            batch_pose = torch.concatenate([global_orient, body_pose], dim=1)
            poses_list.append(batch_pose.detach())
            transl_list.append(transl.detach())
            
        output = {
            'poses': torch.concatenate(poses_list),
            'trans': torch.concatenate(transl_list),
            'joints': torch.concatenate(joint_list)
        }
        return output

class Moshpp(torch.nn.Module):
    def __init__(self,
                 frames_stage1 = 12,
                 frames_stage2 = 100,
                 iter_stage1 = 100,
                 iter_stage2 = 1000,
                 num_betas=10,
                 device='cuda'):
        super().__init__()
        self.frames_stage1 = frames_stage1
        self.frames_stage2 = frames_stage2
        self.iter_stage1 = iter_stage1
        self.iter_stage2 = iter_stage2
        self.device = device
        self.num_betas = num_betas



    def fit(self, input):

        # sample the data for 1st stage
        markers_pos = input['markers_pos']
        markers_pos = markers_pos.to(self.device)
        id = input['id']
        stage1fitter = StageOneFitter(num_frames = self.frames_stage1, num_betas=self.num_betas, num_epochs = self.iter_stage1)
        id_stage1 = torch.randperm(markers_pos.shape[0])[:self.frames_stage1]
        data_stage1 = markers_pos[id_stage1]

        m, betas = stage1fitter.fit(data_stage1, id, num_epochs = self.iter_stage1)

        stage2fitter = StageTwoFitter(num_frames=self.frames_stage2,
                                      num_betas=self.num_betas)
        output = stage2fitter.fit(markers_pos, m, betas, num_epochs = self.iter_stage2)

        output['betas'] = betas
        output['marker_pos'] = m
        
        return output   


class LinearLayers(torch.nn.Module):
    """
    One or multiple dense layers with skip connections from input to final output.
    """

    def __init__(self, hidden_size, num_layers=2, dropout_p=0.0, use_skip=False, use_batch_norm=True):
        super(LinearLayers, self).__init__()
        self.hidden_size = hidden_size

        layers = []
        for _ in range(num_layers):
            new_layers = [torch.nn.Linear(hidden_size, hidden_size)]
            if use_batch_norm:
                bn = torch.nn.BatchNorm1d(hidden_size)
                torch.nn.init.uniform_(bn.weight)
                new_layers.append(bn)
            new_layers.append(torch.nn.PReLU())
            new_layers.append(torch.nn.Dropout(dropout_p))
            layers.extend(new_layers)

        self.layers = torch.nn.Sequential(*layers)

        if use_skip:
            self.skip = lambda x, y: x + y
        else:
            self.skip = lambda x, y: y

    def forward(self, x):
        y = self.layers(x)
        out = self.skip(x, y)
        return out

class FeedForwardResidualBlock(torch.nn.Module):
    """One residual block."""

    def __init__(self, input_size, output_size):
        super(FeedForwardResidualBlock, self).__init__()
        self.dense = torch.nn.Linear(input_size, output_size)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        y = self.dense(x)
        y = y + x
        y = self.activate(y)
        return y

class MLP(torch.nn.Module):
    """
    An MLP mapping from input size to output size going through n hidden dense layers. Uses batch normalization,
    PReLU and can be configured to apply dropout.
    """

    def __init__(self, input_size, output_size, hidden_size, num_layers=2, dropout_p=0.0, skip_connection=False,
                 use_batch_norm=True):
        super(MLP, self).__init__()
        self.input_to_hidden = torch.nn.Linear(input_size, hidden_size)
        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(hidden_size)
            torch.nn.init.uniform_(self.batch_norm.weight)
        else:
            self.batch_norm = torch.nn.Identity()
        self.activation_fn = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.hidden_to_output = torch.nn.Linear(hidden_size, output_size)
        hidden_layers = []
        for _ in range(num_layers):
            h = LinearLayers(hidden_size, dropout_p=dropout_p, use_batch_norm=use_batch_norm, use_skip=skip_connection)
            hidden_layers.append(h)
        self.hidden_layers = torch.nn.Sequential(*hidden_layers)

    def forward(self, x):
        y = self.input_to_hidden(x)
        y = self.batch_norm(y)
        y = self.activation_fn(y)
        y = self.dropout(y)
        y = self.hidden_layers(y)
        y = self.hidden_to_output(y)
        return y


class FrameModel(torch.nn.Module):
    """A frame-wise feed-forward model with residual connections, similar to Holden's "Robust Denoising". """

    def __init__(self,
                 input_size,
                 betas_size,
                 poses_size,
                 trans_size,
                 num_layers,
                 hidden_size,
                 m_dropout = 0,
                 only_pose = False):
        super(FrameModel, self).__init__()
        self._init_args = dict(
            input_size=input_size,
            betas_size=betas_size,
            poses_size=poses_size,
            trans_size=trans_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            m_dropout=m_dropout,
            only_pose=only_pose)
        self.input_size = input_size
        self.betas_size = betas_size
        self.poses_size = poses_size
        self.trans_size = trans_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.only_pose = only_pose
        

        self.from_input = torch.nn.Linear(self.input_size, self.hidden_size)
        self.blocks = torch.nn.Sequential(
            *[FeedForwardResidualBlock(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)])

        self.to_pose = torch.nn.Linear(hidden_size, poses_size)
        self.to_tran = torch.nn.Linear(hidden_size, trans_size)

        self.to_shape = MLP(input_size=self.hidden_size,
                                output_size=betas_size, hidden_size=hidden_size,
                                num_layers=2, dropout_p=m_dropout,
                                skip_connection=False, use_batch_norm=False)


    def model_name(self):
        base_name = "ResNet-{}x{}".format(self.num_layers, self.hidden_size)
        return base_name

    def forward(self, x):
        

        # Estimate pose.
        x = self.from_input(x)
        x = self.blocks(x)
        
        pose_hat = self.to_pose(x)
        tran_hat = self.to_tran(x)
        if self.only_pose:
            shape_hat = torch.zeros([x.shape[0], 1, self.betas_size], device=x.device)
        else:
            s = self.to_shape(x)
            shape_hat = torch.mean(s, dim=1, keepdim=True)

        return {'poses': pose_hat,
                'betas': shape_hat,
                'trans':tran_hat}

    def clone(self):
        # 记录原模型的 device
        device = next(self.parameters()).device

        # 创建新的同构模型
        new_model = type(self)(**self._init_args).to(device)

        # 拷贝参数
        new_model.load_state_dict(copy.deepcopy(self.state_dict()))
        return new_model


class SequenceModel(torch.nn.Module):

    def __init__(self,
                 input_size,
                 betas_size,
                 poses_size,
                 trans_size,
                 num_layers,
                 hidden_size,
                 m_dropout = 0,
                 model_type = 'gru',
                 m_bidirectional= True,
                 only_pose = False):
        super(SequenceModel, self).__init__()
        self.input_size = input_size
        self.betas_size = betas_size
        self.poses_size = poses_size
        self.trans_size = trans_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self._init_args = dict(
            input_size=input_size,
            betas_size=betas_size,
            poses_size=poses_size,
            trans_size=trans_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            m_dropout=m_dropout,
            m_bidirectional=m_bidirectional,
            model_type=model_type,
            only_pose=only_pose
        )
        self.is_bidirectional = m_bidirectional
        self.num_directions = 2 if m_bidirectional else 1
        self.model_type = model_type
        self.learn_init_state = True
        self.only_pose = only_pose


        if m_dropout > 0.0:
            self.input_drop = torch.nn.Dropout(p=m_dropout)
        else:
            self.input_drop = torch.nn.Identity()

        self.init_state = None
        self.final_state = None
        if model_type != 'cnn' and self.learn_init_state:
            self.to_init_state_h = torch.nn.Linear(self.input_size, self.hidden_size * self.num_layers * self.num_directions)
            self.to_init_state_c = torch.nn.Linear(self.input_size, self.hidden_size * self.num_layers * self.num_directions)

        if model_type == 'lstm':
            self.model = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=self.is_bidirectional, batch_first=True)
        elif model_type == 'gru':
            self.model = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers, bidirectional=self.is_bidirectional, batch_first=True)
        elif model_type == 'rnn':
            self.model = torch.nn.RNN(self.input_size, self.hidden_size, self.num_layers, bidirectional=self.is_bidirectional, batch_first=True)
        elif model_type == 'cnn':
            kernel_size = 61
            conv_layers = []
            # 第一层卷积
            conv = torch.nn.Conv1d(
                    in_channels=self.input_size,
                    out_channels=self.hidden_size,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size-1)//2  # 保持时序长度不变
                )
            bn = torch.nn.BatchNorm1d(self.hidden_size)
            act = torch.nn.ReLU()
            conv_layers.extend([conv, bn, act])
            # 剩余卷积层
            for i in range(num_layers-1):
                conv = torch.nn.Conv1d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size-1)//2  # 保持时序长度不变
                )
                bn = torch.nn.BatchNorm1d(self.hidden_size)
                act = torch.nn.ReLU()
                conv_layers.extend([conv, bn, act])
            self.model = torch.nn.Sequential(*conv_layers)
            self.num_directions = 1
        elif model_type == 'transformer':
            self.pos_encoder = torch.nn.Parameter(torch.zeros(1, 300, self.hidden_size))  # 假设最大序列长度512
            torch.nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
            input_proj = torch.nn.Linear(self.input_size, self.hidden_size)
            # Transformer Encoder
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=4,
                dim_feedforward=self.hidden_size * 2,
                dropout=m_dropout,
                activation='relu',
                batch_first=True
            )
            self.transformer_encoder = torch.nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.num_layers
            )
            self.model = torch.nn.Sequential(input_proj, self.transformer_encoder)
            self.num_directions = 1

            
        self.to_pose = torch.nn.Linear(hidden_size * self.num_directions, poses_size)
        self.to_tran = torch.nn.Linear(hidden_size * self.num_directions, trans_size)
        self.to_shape = MLP(input_size=hidden_size * self.num_directions,
                            output_size=betas_size, hidden_size=hidden_size,
                            num_layers=2, dropout_p=m_dropout,
                            skip_connection=False, use_batch_norm=False)

    def cell_init(self, inputs_):
        """Return the initial state of the cell."""
        if self.learn_init_state:
            # Learn the initial state based on the first frame.
            if self.model_type == 'lstm':
                c0 = self.to_init_state_c(inputs_[:, 0:1])
                c0 = c0.reshape(-1, self.num_layers*self.num_directions, self.hidden_size).transpose(0, 1)
                h0 = self.to_init_state_h(inputs_[:, 0:1])
                h0 = h0.reshape(-1, self.num_layers*self.num_directions, self.hidden_size).transpose(0, 1)
                return c0.contiguous(), h0.contiguous()
            else :                
                h0 = self.to_init_state_h(inputs_[:, 0:1])
                h0 = h0.reshape(-1, self.num_layers*self.num_directions, self.hidden_size).transpose(0, 1)
                return h0.contiguous()
        else:
            return self.init_state

    def model_name(self):
        """A summary string of this model."""
        base_name = "Sequence-{}".format('-'.join([str(self.hidden_size)] * self.num_layers))
        base_name = base_name + "-" + self.model_type
        if self.is_bidirectional:
            base_name = "Bi" + base_name
        return base_name

    def forward(self, x, is_new_sequence=True):
        if is_new_sequence:
            self.final_state = None
        self.init_state = self.final_state

        inputs_ = self.input_drop(x)


        # Feed it to the LSTM.
        # Disable CuDNN for higher-order gradients (meta-learning compatibility)
        # with torch.backends.cudnn.flags(enabled=False):
        if self.model_type == 'cnn':
            inputs_ = inputs_.transpose(1, 2)
            out = self.model(inputs_)
            out = out.transpose(1, 2)
        elif self.model_type == 'transformer':
            out = self.model(inputs_)
        else:
            # Get the initial state of the recurrent cells.
            self.init_state = self.cell_init(inputs_)
            out, final_state = self.model(inputs_, self.init_state)
            self.final_state = final_state  


        pose_hat = self.to_pose(out)  # (N, F, self.output_size)

        tran_hat = self.to_tran(out)
        # Estimate shape if configured.

        if self.only_pose:
            shape_hat = torch.zeros([x.shape[0], 1, self.betas_size], device=x.device)
        else:
            s = self.to_shape(out)  # (N, F, N_BETAS)
            shape_hat = torch.mean(s, dim=1, keepdim=True)

        return {'poses': pose_hat,
                'betas': shape_hat,
                'trans': tran_hat}

    def clone(self):
        # 记录原模型的 device
        device = next(self.parameters()).device

        # 创建新的同构模型
        new_model = type(self)(**self._init_args).to(device)

        # 拷贝参数
        new_model.load_state_dict(copy.deepcopy(self.state_dict()))
        return new_model



def main():

    # 1. load data
    cfg = OmegaConf.load('./config.yaml')

    dataset = MocapSequenceDataset(cfg.paths.dataset_dir)
    subjects = set(dataset.subject_names)
    subject_ids = {s:[i for i, n in enumerate(dataset.subject_names) if n == s] for s in subjects}



    for ind in range(len(dataset)):
        stage2_data = dataset[ind]

        # 2. load model
        moshpp = Moshpp(iter_stage1 = 1000, iter_stage2 = 1000)
        stage2_data['markers_pos'] = torch.from_numpy(stage2_data['markers_pos'] / 1000).float()

        moshpp.fit(stage2_data)



    # 3. define loss and training (stage i and ii)

    for data in dataset:
        pass

    # 4. test model



if __name__ == '__main__':
    main()