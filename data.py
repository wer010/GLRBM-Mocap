import os.path as osp
from glob import glob
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import torch
import random
import pandas as pd
from smpl import Smpl
import pickle
from tqdm import tqdm
from geo_utils import estimate_lcs_with_faces
from utils import visualize, visualize_aitviewer
from collections import defaultdict
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_quaternion, quaternion_to_axis_angle, quaternion_multiply, quaternion_invert
import lmdb

rigidbody_marker_id = {
    "head": 335,
    "chest": 3073,
    "left_arm": 2821,
    "left_forearm": 1591,
    "left_hand": 2000,
    "left_leg": 981,
    "left_shin": 1115,
    "left_foot": 3341,
    # "left_hip":809,
    "right_arm": 4794,
    "right_forearm": 5059,
    "right_hand": 5459,
    "right_leg": 4465,
    "right_shin": 4599,
    "right_foot": 6742,
    # "right_hip":4297
}
rbm_parent = [1,-1,1,2,3,1,5,6,
                1,8,9,1,11,12]
moshpp_marker_id = {
    "ARIEL": 411,
    "C7": 3470,
    "CLAV": 3171,
    "LANK": 3327,
    "LBHD": 182,
    "LBSH": 2940,
    "LBWT": 3122,
    "LELB": 1666,
    "LELBIN": 1725,
    "LFHD": 0,
    "LFIN": 2174,
    "LFRM": 1568,
    "LFSH": 1317,
    "LFWT": 857,
    "LHEE": 3387,
    "LIWR": 2112,
    "LKNE": 1053,
    "LKNI": 1058,
    "LMT1": 3336,
    "LMT5": 3346,
    "LOWR": 2108,
    "LSHN": 1082,
    "LTHI": 1454,
    "LTHMB": 2224,
    "LTOE": 3233,
    "LUPA": 1443,
    "MBWT": 3022,
    "MFWT": 3503,
    "RANK": 6728,
    "RBHD": 3694,
    "RBSH": 6399,
    "RBWT": 6544,
    "RELB": 5135,
    "RELBIN": 5194,
    "RFHD": 3512,
    "RFIN": 5635,
    "RFRM": 5037,
    "RFSH": 4798,
    "RFWT": 4343,
    "RHEE": 6786,
    "RIWR": 5573,
    "RKNE": 4538,
    "RKNI": 4544,
    "RMT1": 6736,
    "RMT5": 6747,
    "ROWR": 5568,
    "RSHN": 4568,
    "RTHI": 4927,
    "RTHMB": 5686,
    "RTOE": 6633,
    "RUPA": 4918,
    "STRN": 3506,
    "T10": 3016,
}

def rela_x_fn(x, marker_type = 'moshpp'):
    if marker_type == 'moshpp':
        x_c = torch.mean(x, dim = -2, keepdim=True)
        x_rela = x - x_c
        ret = torch.concatenate([x_c, x_rela], dim = -2)
    elif marker_type == 'rbm':
        x_pos = x[..., :3]
        x_ori = x[..., 3:]
        pos_center = torch.mean(x_pos, dim = -2, keepdim=True)
        rel_pos = x_pos - pos_center

        quat_ori = axis_angle_to_quaternion(x_ori)
        rbm_parent_tensor = torch.tensor(rbm_parent, dtype=torch.int).to(x.device)
        quat_ori_parent = quat_ori[..., rbm_parent_tensor,:]
        identity = quat_ori_parent.new_tensor([1, 0, 0, 0])
        quat_ori_parent[...,1,:] = identity
        rel_quat_ori = quaternion_multiply(quaternion_invert(quat_ori_parent), quat_ori)
        rel_ori = quaternion_to_axis_angle(rel_quat_ori)

        ret = torch.concatenate([pos_center, rel_pos, rel_ori], dim = -2)
    return ret

def extractwindow(sample, window_size=120, mode='random'):
    """
    Extract a window of a fixed size. If the sequence is shorter than the desired window size it will return the
    entire sequence without any padding.
    """
    n_frames = sample['poses'].shape[0]
    assert n_frames >= window_size
    if n_frames==window_size:
        return sample
    else:
        if mode == 'beginning':
            sf, ef = 0, window_size
        elif mode == 'middle':
            mid = n_frames // 2
            sf = mid - window_size // 2
            ef = sf + window_size
        elif mode == 'random':
            sf = torch.randint(0, n_frames - window_size + 1, [1])
            ef = sf + window_size
        else:
            raise ValueError(f"Mode '{mode}' for window extraction unknown.")
        ret = {k:v[sf:ef] for k,v in sample.items() if k != 'betas'}
        ret['betas'] = sample['betas']
        return ret


def train_collate_fn(batch):
    extracted_batch = [extractwindow(sample, 120, mode='random') for sample in batch]
    ret = {k:torch.stack([sample[k] for sample in extracted_batch]) for k in extracted_batch[0].keys()}
    return ret

class AmassLmdbDataset(Dataset):
    def __init__(self, path, use_rela_x = True, marker_type = 'moshpp', device='cuda'):
        super(AmassLmdbDataset, self).__init__()
        self.path = path
        self.use_rela_x = use_rela_x
        self.marker_type = marker_type
        self.device = device
        env = lmdb.open(path, readonly=True)
        with env.begin(write=False) as txn:
            self.len = int(txn.get('__len__'.encode()))
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        env = lmdb.open(self.path, readonly=True, lock=False)
        key = idx.to_bytes(4, byteorder='big')
        with env.begin(write=False) as txn:
            value = txn.get(key)
        data = pickle.loads(value)
        if self.marker_type == 'rbm':
            marker_info = torch.cat([data['marker_pos'], data['marker_ori']], dim=-1)
        else:
            marker_info = data['marker_pos']
        if self.use_rela_x:
            marker_info = rela_x_fn(marker_info, self.marker_type)
        ret = {
            'marker_info': marker_info.to(self.device),
            'poses': data['poses'].to(self.device),
            'betas': data['betas'].to(self.device),
            'trans': data['trans'].to(self.device),
            'joints': data['joints'].to(self.device),
        }
        return ret


def subsample_to_60fps(orig_ft, orig_fps):
        '''Get features at 30fps frame-rate
        Args:
            orig_ft <array> (T, 25*3): Feats. @ `orig_fps` frame-rate
            orig_fps <float>: Frame-rate in original (ft) seq.
        Return:
            ft <array> (T', 25*3): Feats. @ 30fps
        '''
        T  = orig_ft['poses'].shape[0]
        out_fps = 60.0
        # Matching the sub-sampling used for rendering
        if int(orig_fps)%int(out_fps):
            sel_fr = np.floor(orig_fps / out_fps * np.arange(int(out_fps))).astype(int)
            n_duration = int(T/int(orig_fps))
            t_idxs = []
            for i in range(n_duration):
                t_idxs += list(i * int(orig_fps) + sel_fr)
            if int(T % int(orig_fps)):
                last_sec_frame_idx = n_duration*int(orig_fps)
                t_idxs += [x+ last_sec_frame_idx for x in sel_fr if x + last_sec_frame_idx < T ]
        else:
            t_idxs = np.arange(0, T, orig_fps/out_fps, dtype=int)

        ft = {}
        for key, values in orig_ft.items():
            if key in ['poses', 'trans']:
                ft[key] = values[t_idxs]
            else:
                ft[key] = values

        return ft


def convert_amass_to_lmdb(data_root, output_file, marker_type):
    """Convert AMASS to LMDB format that we can use during training."""
    if marker_type == 'rbm':
        vid = [value for value in rigidbody_marker_id.values()]
    elif marker_type == 'moshpp':
        vid = [value for value in moshpp_marker_id.values()]
    n_marker = len(vid)
    device = 'cuda'
    pos_offset = torch.tensor([0.0095, 0, 0, 1]).expand([n_marker, -1]).to(device)
    ori_offset = torch.eye(3).expand([n_marker, -1, -1]).to(device)
    vid_tensor = torch.tensor(vid).to(device)

    output_file = output_file + '_' + marker_type

    print("Converting AMASS data under {} and exporting it to {} ...".format(data_root, output_file))

    npz_file_ids = glob(osp.join(data_root, '*',  '*stageii.npz'))

    env = lmdb.open(output_file, map_size=10*1024**3)
    with env.begin(write=True) as txn:
        valid_i = 0
        for file_id in tqdm(npz_file_ids):
            sample = np.load(osp.join(file_id),allow_pickle=True)
            fps = sample['mocap_frame_rate']
            if fps <60:
                print(f'{file_id} has fps < 60, skip.')
                continue

            if isinstance(sample['betas'],list):
                print('only single person data is supported.')
                continue
            sample = subsample_to_60fps(sample, fps)
            if sample['poses'].shape[0] <120:
                continue
            poses_body = sample['poses'][:, :66]  # (N_FRAMES, 66)
            poses_hands = np.zeros([poses_body.shape[0], 6])
            poses = np.concatenate([poses_body, poses_hands],axis=-1)
            # 转成torch tensor
            poses = torch.from_numpy(poses).float().to(device)
            betas = torch.from_numpy(sample['betas'][:10]).float().to(device)  # (N_SHAPE_PARAMS, )
            trans = torch.from_numpy(sample['trans']).float().to(device)  # (N_FRAMES, 30)
            

            # Extract joint information, watch out for CUDA out of memory.
            n_frames = poses.shape[0]
            
            if n_frames>3000:
                n_slices = n_frames//3000
                joints_list = []
                marker_pos_list = []
                marker_ori_list = []
                for i in range(n_slices+1):
                    marker_pos, marker_ori, v_posed, joints = virtual_marker(betas,
                                                                     poses[3000*i:3000*(i+1),:],
                                                                     trans[3000*i:3000*(i+1),:],
                                                                     vid_tensor,
                                                                     pos_offset,
                                                                     ori_offset,
                                                                     visualize_flag=False)

                    marker_pos_list.append(marker_pos)
                    marker_ori_list.append(marker_ori)
                    joints_list.append(joints)
                joints = torch.concatenate(joints_list)
                marker_pos = torch.concatenate(marker_pos_list)
                marker_ori = torch.concatenate(marker_ori_list)
            else:
                marker_pos, marker_ori, v_posed, joints = virtual_marker(betas,
                                                                     poses,
                                                                     trans,
                                                                     vid_tensor,
                                                                     pos_offset,
                                                                     ori_offset,
                                                                     visualize_flag=False)


            assert joints.shape[0] == n_frames
            marker_ori = matrix_to_axis_angle(marker_ori)

            data_dict = {
                    'poses': poses.detach().cpu(),     # (T, J*3)
                    'betas': betas.detach().cpu(),     # (10,) or (16,)
                    'trans': trans.detach().cpu(),     # (T, 3)
                    'joints': joints.detach().cpu(),            # 可选字段
                    'n_frames': n_frames,
                    'marker_pos': marker_pos.detach().cpu(),
                    'marker_ori': marker_ori.detach().cpu(),
                    'file_path': file_id      # 保留原始路径，便于调试
                }
                
            # 使用pickle序列化整个字典
            key = valid_i.to_bytes(4, byteorder='big')
            value = pickle.dumps(data_dict, protocol=4)  # protocol=4更高效
            txn.put(key, value)
            valid_i += 1


        txn.put('__len__'.encode(), "{}".format(valid_i).encode())



def virtual_marker(betas, 
                   pose, 
                   trans, 
                   vid, 
                   pos_offset=None, 
                   ori_offset=None,
                   visualize_flag = False):
    # get the 6 dof info of markers under the given pose
    '''
    :param betas: shape (10)
    :param pose: shape (n,24*3)
    :param trans: shape (n,3)
    :param vid: shape (m) m is the num of markers
    :param pos_offset: (m, 4)
    :param ori_offset: (m, 3, 3)
    :return:
    '''
    model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz', device=betas.device)
    output = model(betas=betas,
                     body_pose=pose[:,3:],
                     global_orient=pose[:,0:3],
                     transl=trans)
    v_posed = output['vertices']
    joints = output['joints']

    vid_tensor = vid[None,:].expand([pose.shape[0], -1]).to(betas.device)
    lcs = estimate_lcs_with_faces(vid=vid_tensor,
                                  fid=model.vertex_faces[vid_tensor],
                                  vertices=v_posed,
                                  faces=model.faces_tensor)

    marker_pos = torch.matmul(lcs, pos_offset[None, ..., None])[:, :, 0:3, 0]
    marker_ori = torch.matmul(lcs[:, :, 0:3, 0:3], ori_offset)
    if visualize_flag:
        # visualize a random frame
        i = torch.randint(0, v_posed.shape[0], [])
        visualize(v_posed[i].cpu().detach().numpy(), model.faces,
                [joints[i].cpu().detach().numpy()], lcs[i].cpu().detach().numpy())
        visualize_aitviewer('smpl',
                            full_poses=pose,
                            betas=betas,
                            trans=trans,
                            extra_points=[marker_pos.detach().cpu().numpy()])

    return marker_pos, marker_ori, v_posed, joints


if __name__ == '__main__':
    pass
    # convert_amass_to_lmdb(data_root='data/CMU', output_file='data/CMU_lmdb', marker_type='rbm')
    # convert_amass_to_lmdb(data_root='data/CMU', output_file='data/CMU_lmdb', marker_type='moshpp')

    # convert_amass_to_lmdb(data_root='data/BMLrub', output_file='data/BMLrub_lmdb', marker_type='rbm')
    # convert_amass_to_lmdb(data_root='data/BMLrub', output_file='data/BMLrub_lmdb', marker_type='moshpp')
    