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


class MetaBabelDataset(Dataset):
    # Dataset class for meta training, the getitem func return data of a task
    def __init__(self, path, use_rela_x = False, marker_type = 'moshpp', device = 'cuda'):
        self.use_rela_x = use_rela_x
        self.marker_type = marker_type

        self.device = device
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.task_id = {}
        self.samples_per_task = []
        for i, key in enumerate(self.data.keys()):
            self.task_id[i] = key
            

    def __len__(self):
        return len(self.task_id)

    def __getitem__(self, t):
        data = self.data[self.task_id[int(t)]]
        data = {key:value.to(self.device) for key, value in data.items()}
        data['task_name'] = self.task_id[int(t)]
        if self.use_rela_x:
            data['marker_info'] = rela_x_fn(data['marker_info'], self.marker_type)
        return data


class BabelDataset(Dataset):
    def __init__(self, path, use_rela_x = False, marker_type = 'moshpp', device='cuda'):
        self.use_rela_x = use_rela_x
        self.marker_type = marker_type
        self.device = device
        with open(path, 'rb') as f:
            raw_data = pickle.load(f)
        fv = next(iter(raw_data.values()))
        all_data = {key:[] for key in fv.keys()}

        for v in raw_data.values():
            for key, value in v.items():
                all_data[key].append(value)

        self.data = {k:torch.concatenate(v) for k,v in all_data.items()}

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, t):
        data = {key:value[t].to(self.device) for key, value in self.data.items()}
        if self.use_rela_x:
            data['marker_info'] = rela_x_fn(data['marker_info'], self.marker_type)
        return data



class MetaCollate:
    def __init__(self, support_ratio=0.5, shuffle = False):
        self.support_ratio = support_ratio
        self.shuffle = shuffle

    def __call__(self, data_list):
        

        support_sets = defaultdict(list)
        query_sets = defaultdict(list)

        for item in data_list:
            n = item['poses'].shape[0]
            indices = list(range(n))
            if self.shuffle:
                random.shuffle(indices)

            split = int(n * self.support_ratio)  # 60% support
            support_idx = indices[:split]
            query_idx = indices[split:]

            # 构建 support / query set
            for key in item.keys():
                if key == 'task_name':
                    support_sets[key].append(item[key])
                    query_sets[key].append(item[key])
                else:
                    support_sets[key].append(item[key][support_idx])
                    query_sets[key].append(item[key][query_idx])
        supp_ret = {}
        qry_ret = {}
        for key in support_sets.keys():
            if key != 'task_name':
                supp_ret[key] = torch.stack(support_sets[key])
                qry_ret[key] = torch.stack(query_sets[key])
            else:
                supp_ret[key] = support_sets[key]
                qry_ret[key] = query_sets[key]

        return supp_ret, qry_ret




def generate_marker_data(fp, marker_type = 'rbm', normalize_flags=[]):
    if marker_type == 'rbm':
        vid = [value for value in rigidbody_marker_id.values()]
    elif marker_type == 'moshpp':
        vid = [value for value in moshpp_marker_id.values()]

    n_marker = len(vid)
    device = 'cuda'
    pos_offset = torch.tensor([0.0095, 0, 0, 1]).expand([n_marker, -1]).to(device)
    ori_offset = torch.eye(3).expand([n_marker, -1, -1]).to(device)
    vid_tensor = torch.tensor(vid).to(device)
    dataset = MetaBabelDataset(fp, device=device)
    save_data = {}
    for t in range(len(dataset)):
        print(f'{t}: Generate the marker for {dataset.task_id[int(t)]}')
        data = dataset[t]

        for k in normalize_flags:
            data[k] = data[k]*0.0
        marker_pos_list = []
        marker_ori_list = []
        joints_list = []
        for i in range(len(data['betas'])):
            frame_num = data['poses'][i].shape[0]
            vid_tensor = vid_tensor.expand([frame_num, -1]).to(device)
            marker_pos, marker_ori, v_posed, joints = virtual_marker(data['betas'][i],
                                                                     data['poses'][i],
                                                                     data['trans'][i],
                                                                     vid_tensor,
                                                                     pos_offset,
                                                                     ori_offset,
                                                                     visualize_flag=False)
            marker_pos_list.append(marker_pos)
            marker_ori_list.append(marker_ori)
            joints_list.append(joints)

        marker_pos = torch.stack(marker_pos_list)
        marker_ori = torch.stack(marker_ori_list)
        marker_ori = matrix_to_axis_angle(marker_ori)
        if marker_type == 'rbm':
            marker_info = torch.cat([marker_pos, marker_ori], dim=-1)
        else:
            marker_info = marker_pos
        joints = torch.stack(joints_list)

        ret = {'betas': data['betas'].detach().cpu(),
               'poses': data['poses'].detach().cpu(),
               'trans': data['trans'].detach().cpu(),
               'marker_info': marker_info.detach().cpu(),
               'joints': joints.detach().cpu()}
        save_data[dataset.task_id[int(t)]] = ret

    if len(normalize_flags) > 0:
        s = '_normalize_'+ '_'.join(str(i) for i in normalize_flags)
    else:
        s = ''
    file_name = fp.replace('.pkl', f'_with_{marker_type}{s}_marker.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(save_data, f)
    return file_name

def convert_dataset():
    df = pd.read_excel('/home/lanhai/PycharmProjects/mosr/cmu_motion.xlsx', header = None)

    motion_cates = df.iloc[:,0].dropna().unique().tolist()
    sub_motion_cats = df.iloc[:,1].to_list()
    subject_ids = df.columns[2:]

    all_paths = []
    motions = {}
    for row_idx, row in df.iterrows():

        if not pd.isna(row[0]):
            sub_motions = {}
            motion = row[0]
        sub_motion_key = row[1]
        sub_motion = {}
        for sub in row[2:].dropna().to_list():
            match = re.search(r'Subject (\d+)', sub)
            sub_ind = match.group(1).zfill(2)
            trials_ind_list = re.findall(r'\n(\d+)', sub)
            sub_motion[sub_ind] = [s.zfill(2) for s in trials_ind_list]
        sub_motions[sub_motion_key] = sub_motion
        motions[motion] = sub_motions
    one_motions = {}
    all_motions = []
    for motion, value in motions.items():
        file_paths = []
        for sub_motion, sub_value in value.items():
            for subject, trials in sub_value.items():
                for t in trials:
                    file_path = osp.join('/home/lanhai/restore/dataset/mocap/amass/CMU-smplx-n/', subject, f'{subject}_{t}_stageii.npz')
                    file_paths.append(file_path)
                    trial = {
                        'file_path':file_path,
                        'trial_id': t,
                        'subject_id': subject,
                        'sub_motion':sub_motion,
                        'motion': motion
                    }
                    all_motions.append(trial)
        one_motions[motion] =  file_paths

    model = Smpl(model_path='/home/lanhai/restore/dataset/mocap/models/smpl/SMPL_NEUTRAL.npz', device='cpu')

    for key, value in one_motions.items():
        output_file = f'/home/lanhai/restore/dataset/mocap/amass/CMU_motion/{key}.pkl'
        if osp.exists(output_file):
            continue
        res = {}
        motions = []
        num_trials = len(value)
        valid_num = 0
        for file_path in value:
            if osp.exists(file_path):
                # convert to lmdb
                valid_num+=1

                data = np.load(file_path, allow_pickle=True)
                print(f'{file_path} is {valid_num}th available amass data in {num_trials} trials with {data["poses"].shape[0]} frames')

                betas = torch.from_numpy(data['betas'][0:10]).float()
                # gender = torch.from_numpy(data['genders'][0])
                pose_body = data['poses'][:, 0:66]
                pose_hands = np.zeros([pose_body.shape[0], 6])
                poses = np.concatenate([pose_body,pose_hands],axis=-1)
                poses = torch.from_numpy(poses).float()
                trans = torch.from_numpy(data['trans']).float()
                if data["poses"].shape[0]>3000:
                    n_slices = data["poses"].shape[0]//3000
                    joints_list = []
                    for i in range(n_slices+1):
                        joints_list.append(model(betas=betas,
                                       body_pose=poses[3000*i:3000*(i+1), 3:],
                                       global_orient=poses[3000*i:3000*(i+1), 0:3],
                                       transl=trans[3000*i:3000*(i+1),:])['joints'])
                    joints = torch.concatenate(joints_list)
                else:
                    joints = model(betas=betas,
                              body_pose=poses[:,3:],
                              global_orient=poses[:, 0:3],
                              transl=trans)['joints']
                trial = {'betas': betas.cpu(),
                         'poses': poses.cpu(),
                         'trans': trans.cpu(),
                         'joints': joints.cpu()
                         }
                motions.append(trial)
        print(f'{valid_num} trials is save in {key} motion class')

        with open(output_file,'wb') as f:
            pickle.dump(motions, f)
        del motions
        torch.cuda.empty_cache()
    print('All done')

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
    :param vid: shape (n, m) m is the num of markers
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

    lcs = estimate_lcs_with_faces(vid=vid,
                                  fid=model.vertex_faces[vid],
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
    # convert_dataset()
    marker_type = 'rbm'
    train_file_name = generate_marker_data('/home/lanhai/restore/dataset/mocap/mosr/meta_train_data.pkl', marker_type = marker_type, normalize_flags=['betas'])

    val_file_name = generate_marker_data('/home/lanhai/restore/dataset/mocap/mosr/meta_val_data.pkl', marker_type = marker_type, normalize_flags=['betas'])

    with open(train_file_name, 'rb') as f:
        raw_train_data = pickle.load(f)
    with open(val_file_name, 'rb') as f:
        raw_val_data = pickle.load(f)

    merge_data = {}
    for key, value in raw_train_data.items():
        cache = {}
        for k,v in value.items():
            cache[k] = torch.concatenate([raw_train_data[key][k], raw_val_data[key][k][0:10]])
        merge_data[key] = cache

    keys = list(merge_data.keys())
    first27_keys = keys[:27]
    last3_keys = keys[27:]

    A_first27 = {k: merge_data[k] for k in first27_keys}
    A_last3 = {k: merge_data[k] for k in last3_keys}

    # 保存成 pkl
    with open(f"/home/lanhai/restore/dataset/mocap/mosr/{marker_type}/{osp.basename(train_file_name)}", "wb") as f:
        pickle.dump(A_first27, f)

    with open(f"/home/lanhai/restore/dataset/mocap/mosr/{marker_type}/{osp.basename(val_file_name)}", "wb") as f:
        pickle.dump(A_last3, f)