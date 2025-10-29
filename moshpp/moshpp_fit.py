import multiprocessing
import os
import time
from copy import deepcopy
from datetime import timedelta
import numpy as np
import torch
import sys
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
logger.remove()  # 移除默认 handler
logger.add(sys.stderr, level="INFO")
from tqdm import tqdm
import chumpy as ch
from sklearn.neighbors import NearestNeighbors
from smpl_fast_derivatives import load_moshpp_models
from transformed_lm import TransformedCoeffs, TransformedLms
from data import virtual_marker, moshpp_marker_id, AmassLmdbDataset
from mesh_distance_main import PtsToMesh
from torch.utils.data import DataLoader
from psbody.mesh import Mesh
import cv2
import scipy
from utils import vis_diff_aitviewer
from metric import MetricsEngine
import lmdb
import pickle

def rigid_landmark_transform(a, b):
    """
    Args:
        a: a 3xN array of vertex locations
        b: a 3xN array of vertex locations

    Returns: (R,T) such that R.dot(a)+T ~= b
    Based on Arun et al, "Least-squares fitting of two 3-D point sets," 1987.
    See also Eggert et al, "Estimating 3-D rigid body transformations: a
    comparison of four major algorithms," 1997.
    """
    assert (a.shape[0] == 3)
    assert (b.shape[0] == 3)
    b = np.where(np.isnan(b), a, b)
    a_mean = np.mean(a, axis=1).reshape((-1, 1))
    b_mean = np.mean(b, axis=1).reshape((-1, 1))
    a_centered = a - a_mean
    b_centered = b - b_mean

    c = a_centered.dot(b_centered.T)
    u, s, v = np.linalg.svd(c, full_matrices=False)
    v = v.T
    R = v.dot(u.T)

    if scipy.linalg.det(R) < 0:
        v[:, 2] = -v[:, 2]
        R = v.dot(u.T)

    T = (b_mean - R.dot(a_mean)).reshape((-1, 1))

    return (R, T)

def perform_rigid_adjustment(poses, trans, opt_models, markers_obs, markers_sim):
    for sv_idx, _ in enumerate(opt_models):

        obs_mrk = markers_obs[sv_idx]
        sim_mrk = markers_sim[sv_idx]
        if isinstance(sim_mrk, np.ndarray):
            R, T = rigid_landmark_transform(sim_mrk.T, obs_mrk.T)
        else:
            R, T = rigid_landmark_transform(sim_mrk.r.T, obs_mrk.T)

        poses[sv_idx][:3] = cv2.Rodrigues(R)[0].ravel()
        trans[sv_idx][:] = T.ravel()


def mosh_stagei_chumpy(stagei_frames) -> dict:
    """
    This is supposed to be used for estimation of subject shape from a list of a single subject performed mocap sessions
    it can also be used for fine tuning of the can markers.
    When using hand it is assumed that the finger markers dont move away that far from thir initial guess so can markers upto body markers are updated

    :return:
    """

    betas = None

    # Todo: check for it if given vtempalte the values for betas should be zeros

    # 1. Loading SMPL models.
    # Canonical model is for canonical(a.k.a can space). the beta params of the can_model are ultimately used
    # Optimization models are associated with each frame

    can_model, opt_models = load_moshpp_models()

    optimize_betas = True
    # T pose 下的marker位置
    vid = [value for value in moshpp_marker_id.values()]
    n_marker = len(vid)
    pos_offset = torch.tensor([0.0095, 0, 0, 1]).expand([n_marker, -1])
    ori_offset = torch.eye(3).expand([n_marker, -1, -1])
    vid_tensor = torch.tensor(vid)
    marker_pos, marker_ori, v_posed, joints = virtual_marker(torch.zeros(10),
                                                            torch.zeros(1, 24*3),
                                                            torch.zeros(1, 3),
                                                            vid_tensor,
                                                            pos_offset,
                                                            ori_offset,
                                                            visualize_flag=False)

    markers_latent = ch.array(marker_pos[0].cpu().detach().numpy())
    can_v = can_model.r
    can_mesh = Mesh(v=can_v, f=can_model.f)
    surface_distance = PtsToMesh(
        sample_verts=markers_latent,
        reference_verts=can_model,
        reference_faces=can_mesh.f,
        reference_template_or_sampler=can_mesh,
        rho=lambda x: x,
        normalize=False,  # want in meters, so don't normalize
        signed=True)
    m2b_dist = np.ones(53) * 0.0095
    desired_distances = ch.array(np.array(m2b_dist))
    distance_to_surface_obj = surface_distance - desired_distances

    # marker_vids = {
    #     "ARIEL": 9011,
    #     "C7": 3832,
    #     "CLAV": 5533,
    #     "LANK": 5882,
    #     "LBHD": 2026,
    #     "LBSH": 4137,
    #     "LBWT": 5697,
    #     "LELB": 1666,
    #     "LELBIN": 1725,
    #     "LFHD": 0,
    #     "LFIN": 2174,
    #     "LFRM": 1568,
    #     "LFSH": 1317,
    #     "LFWT": 857,
    #     "LHEE": 3387,
    #     "LIWR": 2112,
    #     "LKNE": 1053,
    #     "LKNI": 1058,
    #     "LMT1": 3336,
    #     "LMT5": 3346,
    #     "LOWR": 2108,
    #     "LSHN": 1082,
    #     "LTHI": 1454,
    #     "LTHMB": 2224,
    #     "LTOE": 3233,
    #     "LUPA": 1443,
    #     "MBWT": 3022,
    #     "MFWT": 3503,
    #     "RANK": 6728,
    #     "RBHD": 3694,
    #     "RBSH": 6399,
    #     "RBWT": 6544,
    #     "RELB": 5135,
    #     "RELBIN": 5194,
    #     "RFHD": 3512,
    #     "RFIN": 5635,
    #     "RFRM": 5037,
    #     "RFSH": 4798,
    #     "RFWT": 4343,
    #     "RHEE": 6786,
    #     "RIWR": 5573,
    #     "RKNE": 4538,
    #     "RKNI": 4544,
    #     "RMT1": 6736,
    #     "RMT5": 6747,
    #     "ROWR": 5568,
    #     "RSHN": 6473,
    #     "RTHI": 6832,
    #     "RTHMB": 7575,
    #     "RTOE": 8481,
    #     "RUPA": 6777,
    #     "STRN": 5531,
    #     "T10": 5623,
    # }
    # latent_labels = list(marker_vids.keys())

    logger.debug(f'Estimating for #latent markers: {len(markers_latent)}')

    tc = TransformedCoeffs(can_body=can_model, markers_latent=markers_latent)
    tc.markers_latent = markers_latent
    tc.can_body = can_model

    # list of chained objects to update estimated markers w.r.t transformed model verts in 12 sample (posed) frames
    markers_sim_all = [TransformedLms(transformed_coeffs=tc, can_body=model) for model in opt_models]

    # Init Markers
    tc2 = TransformedCoeffs(can_body=can_model.r, markers_latent=markers_latent.r)
    init_markers_latent = TransformedLms(transformed_coeffs=tc2, can_body=can_model)
    # todo: couldn't you simply call init_markers_latent = markers_latent.r?

    lm_diffs = []
    markers_sim = []
    markers_obs = []
    labels_obs = []

    for fidx, obs_frame_data in enumerate(stagei_frames):
        # cur_frame = {}
        # common_labels = list(set(latent_labels).intersection(set(obs_labels)))
        obf = obs_frame_data.reshape(-1,3)
        # lm_ids = [latent_labels.index(k) for k in common_labels]
        smf = markers_sim_all[fidx]

        markers_obs.append(obf)
        markers_sim.append(smf)
        lm_diffs.append(obf-smf)

    data_obj = ch.vstack([ld for ld in lm_diffs])

    logger.debug('Number of available markers in each stagei selected frames: {}'.format(
        ', '.join([f'(F{fIdx:02d}, {len(frame)})' for fIdx, frame in enumerate(markers_obs)])))

    # Rigidly adjust pose_ids/trans to fit bodies to landmarks
    logger.debug('Rigidly aligning the body to the markers')
    poses = [model.pose for model in opt_models]
    trans = [model.trans for model in opt_models]
    perform_rigid_adjustment(poses, trans, opt_models, [ob for ob in markers_obs], markers_sim)

    # # Set up objective
    stagei_wts = {
        "stagei_wt_poseH": 3.0,
        "stagei_wt_poseF": 3.0,
        "stagei_wt_expr": 34.0,
        "stagei_wt_pose": 3.0,
        "stagei_wt_poseB": 3.0,
        "stagei_wt_init_finger_left": 400.0,
        "stagei_wt_init_finger_right": 400.0,
        "stagei_wt_init_finger": 400.0,
        "stagei_wt_betas": 10.0,
        "stagei_wt_init": 300,
        "stagei_wt_data": 75.0,
        "stagei_wt_surf": 10000.0,
        "stagei_wt_annealing": [1.0, 0.5, 0.25, 0.125],
    }

    # Setup Variables
    if optimize_betas:
        v_betas = [can_model.betas[:10]]

    all_pose_ids = list(range(can_model.pose.size))
    pose_body_ids = []
    pose_finger_ids = []
    pose_face_ids = []
    pose_root_ids = all_pose_ids[:3]
    pose_body_ids = all_pose_ids[3:66]

    detailed_step = False

    for tidx, wt_anneal_factor in enumerate(stagei_wts['stagei_wt_annealing']):
        if tidx > len(stagei_wts['stagei_wt_annealing']) - 3: detailed_step = True

        opt_objs = {}

        if len(pose_body_ids):
            wt_poseB = stagei_wts['stagei_wt_poseB'] * wt_anneal_factor
        if optimize_betas: wt_beta = stagei_wts['stagei_wt_betas'] * wt_anneal_factor

        wt_data = (stagei_wts['stagei_wt_data'] / wt_anneal_factor) * (46 / 53)

        wt_init = {'body':300.0}
        # wt_surf = {k: stagei_wts.get(f'stagei_wt_surf_{k}', stagei_wts.stagei_wt_surf) for k in
        #            marker_meta['marker_type_mask'].keys()}

        wt_messages = f'Step {tidx + 1}/{len(stagei_wts["stagei_wt_annealing"])} :' \
                      f' Opt. wt_anneal_factor = {wt_anneal_factor:.2f}, ' \
                      f'wt_data = {wt_anneal_factor:.2f}, wt_poseB = {wt_data:.2f}'

        logger.debug(wt_messages)
        logger.debug(
            f'stagei_wt_init for different marker types {", ".join([f"{k} = {v:.02f}" for k, v in wt_init.items()])}: ')
        # logger.debug(
        #     f'stagei_wt_surf for different marker types {", ".join([f"{k} = {v:.02f}" for k, v in wt_surf.items()])}')

        # opt_objs.update({'data_%s' % k: data_obj[k] * wt_data[k] for k in can_meta['mrk_ids']})
        opt_objs['data'] = data_obj * wt_data

        if len(pose_body_ids):
            if opt_models[0].priors['pose']:
                opt_objs['poseB'] = ch.concatenate(
                    [model.priors['pose'](model.pose[pose_body_ids]) for model in opt_models]) * wt_poseB

        init_loss = (markers_latent - init_markers_latent)

        # TODO: 这里没啥具体作用，就是给头部marker加一个权重，可以砍掉，全身都用同一个权重就行

        opt_objs.update({f'init_body': init_loss * wt_init['body']})

        if optimize_betas: opt_objs['beta'] = can_model.priors['betas'].ravel() * wt_beta
        opt_objs['surf'] = distance_to_surface_obj * stagei_wts['stagei_wt_surf']

        free_vars = trans + [markers_latent]
        pose_ids = pose_root_ids + pose_body_ids

        if len(pose_body_ids):
            pose_ids = list(set(pose_ids).difference(set(all_pose_ids[30:36])))

        pose_ids = sorted(list(set(pose_ids)))
        v_poses = [model.pose[pose_ids] for model in opt_models]

        free_vars += v_poses
        if optimize_betas: free_vars += v_betas

        logger.debug("Init. loss values: {}".format(' | '.join(
            [f'{k} = {np.sum(opt_objs[k].r ** 2):2.2e}' for k in sorted(opt_objs)])))
        ch.minimize(fun=opt_objs, x0=free_vars,
                    callback=None,
                    method='dogleg',
                    options={'e_3': 0.001,
                             'delta_0': .5, 'disp': None,
                             'maxiter': 100})
        logger.debug("Final loss values: {}".format(' | '.join(
            [f'{k} = {np.sum(opt_objs[k].r ** 2):2.2e}' for k in sorted(opt_objs)])))

    # Saving all values from the optimization process, (sum of least squares)
    stagei_errs = {k: np.sum(obj_val.r ** 2) for k, obj_val in opt_objs.items()}

    sknbrs = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(can_model.r)
    _, closest = sknbrs.kneighbors(markers_latent.r)
    markers_latent_vids = [el[0] for el in closest.tolist()]


    stagei_data = {'betas': can_model.betas.r if hasattr(can_model, 'betas') else None,
                   'markers_latent': markers_latent.r,
                   'markers_latent_vids': markers_latent_vids, }

    return stagei_data


def mosh_stageii_chumpy(x, stagei_data) -> dict:
    num_train_markers = 46  # constant

    # 1. Load observed markers
    markers_latent=stagei_data['markers_latent']
    # latent_labels=stagei_data['latent_labels']
    betas=stagei_data['betas']

    logger.debug('Loaded mocap markers for mosh stageii')

    # avail_labels = latent_labels

    can_model, opt_models = load_moshpp_models(num_beta_shared_models=1)

    opt_model = opt_models[0]

    if hasattr(can_model, 'betas'):
        can_model.betas[:10] = betas[:10].copy()

    tc = TransformedCoeffs(can_body=can_model.r, markers_latent=markers_latent)
    markers_sim_all = TransformedLms(transformed_coeffs=tc, can_body=opt_model)

    # logger.debug(f'#observed, #simulated markers: {len(mocap.labels)}, {len(markers_sim_all)}')

    on_step = None

    perframe_data = {
        'markers_sim': [],
        'markers_obs': [],
        'labels_obs': [],
        'fullpose': [],
        'trans': [],
        'stageii_errs': {},
    }

    logger.debug(
        'mosh stageii weights are subject to change during the optimization, depending on how many markers are absent in each frame.')

    stageii_wts = {
        "stageii_wt_data": 400,
        "stageii_wt_velo": 2.5,
        "stageii_wt_dmpl": 1.0,
        "stageii_wt_expr": 1.0,
        "stageii_wt_poseB": 1.6,
        "stageii_wt_poseH": 1.0,
        "stageii_wt_poseF": 1.0,
        "stageii_wt_annealing": 2.5,
    }
    logger.debug(
        "MoSh stagei weights before annealing:\n{}".format(
            "\n".join(
                [
                    "{}: {}".format(k, wt)
                    for k, wt in stageii_wts.items()
                    if k.startswith("stageii_wt")
                ]
            )
        )
    )

    selected_frames = range(0, x.shape[0])
    logger.debug(f'Starting mosh stageii for {len(selected_frames)} frames.')

    pose_prev = None
    dmpl_prev = None

    # Setup Variables

    all_pose_ids = list(range(opt_model.pose.size))
    pose_root_ids = all_pose_ids[:3]

    pose_body_ids = all_pose_ids[3:66]
    
    first_active_frame = True
    observed_markers_dict = x

    for fIdx, t in enumerate(selected_frames):

        if len(observed_markers_dict[t]) == 0:
            logger.error(f'no available observed markers for frame {t}. skipping the frame.')
            continue

        # Todo: should markers_obs be chumpy array?
        markers_obs = observed_markers_dict[t].reshape(-1,3)
        markers_sim = markers_sim_all

        anneal_factor = 1.

        wt_data = stageii_wts['stageii_wt_data'] * (num_train_markers / markers_obs.shape[0])
        wt_pose = stageii_wts['stageii_wt_poseB'] * anneal_factor
        wt_poseH = stageii_wts['stageii_wt_poseH'] * anneal_factor
        wt_poseF = stageii_wts['stageii_wt_poseF'] * anneal_factor
        wt_dmpl = stageii_wts['stageii_wt_dmpl']
        wt_expr = stageii_wts['stageii_wt_expr']
        wt_velo = stageii_wts['stageii_wt_velo']

        # Setting up objective
        opt_objs = {'data': (markers_sim - markers_obs) * wt_data}
        if len(pose_body_ids):
            opt_objs['poseB'] = opt_model.priors['pose'](opt_model.pose[pose_body_ids]) * wt_pose


        if pose_prev is not None:
            # extrapolating from prev 2 frames
            opt_objs['velo'] = (opt_model.pose - (opt_model.pose.r + (opt_model.pose.r - pose_prev))) * wt_velo

        # 1. Fit only the first frame
        if first_active_frame:  # np.median(np.abs(data_obj.r.ravel())) > .03:
            # Rigidly adjust pose_ids/trans to fit bodies to landmarks
            logger.debug('Rigidly aligning the markers to the body...')
            # opt_model.pose[:] = opt_model.pose.r
            # opt_model.trans[:] = opt_model.trans.r
            perform_rigid_adjustment([opt_model.pose], [opt_model.trans], [opt_model], [markers_obs], [markers_sim])

            # for wt_pose_first in [5.]:
            for wt_pose_first in [10. * wt_pose, 5. * wt_pose, wt_pose]:
                if len(pose_body_ids):
                    opt_objs['poseB'] = opt_model.priors['pose'](opt_model.pose[pose_body_ids]) * wt_pose_first
                
                pose_ids = pose_root_ids + pose_body_ids
                if len(pose_body_ids):
                    pose_ids = list(set(pose_ids).difference(set(all_pose_ids[30:36])))

                free_vars = [opt_model.trans, opt_model.pose[pose_ids]]

                ch.minimize(fun= opt_objs, x0=free_vars,
                            method='dogleg',
                            options={'e_3': .001, 'delta_0': 5e-1, 'disp': None, 'maxiter':100})

            first_active_frame = False
        else:
            pose_prev = opt_model.pose.r.copy()


        # 1. Warm start to correct pose
        logger.debug(
            f'{fIdx:04d}/{len(selected_frames):04d} -- Step 1. initial loss values: {" | ".join(["{} = {:2.2e}".format(k, np.sum(v.r ** 2)) for k, v in opt_objs.items()])}')

        pose_ids = pose_root_ids + pose_body_ids
        if len(pose_body_ids):
            pose_ids = list(set(pose_ids).difference(set(all_pose_ids[30:36])))
        free_vars = [opt_model.trans, opt_model.pose[pose_ids]]
        ch.minimize(fun=opt_objs, x0=free_vars,
                    method='dogleg',
                    options={'e_3': .01, 'delta_0': 5e-1, 'disp': None, 'maxiter': 100})
        logger.debug(
            f'{fIdx:04d}/{len(selected_frames):04d} -- Step 1. final loss values: {" | ".join(["{} = {:2.2e}".format(k, np.sum(v.r ** 2)) for k, v in opt_objs.items()])}')

        # 2. Fit for full pose
        free_vars = [opt_model.trans]
        pose_ids = pose_root_ids + pose_body_ids
        if len(pose_body_ids):
            pose_ids = list(set(pose_ids).difference(set(all_pose_ids[30:36])))


        pose_ids = sorted(list(set(pose_ids)))
        free_vars += [opt_model.pose[pose_ids]]

        logger.debug(
            f'{fIdx:04d}/{len(selected_frames):04d} -- Step 2. initial loss values: {" | ".join(["{} = {:2.2e}".format(k, np.sum(v.r ** 2)) for k, v in opt_objs.items()])}')
        ch.minimize(fun=opt_objs, x0=free_vars,
                    method='dogleg',
                    options={'e_3': .01, 'delta_0': 5e-1, 'disp': None, 'maxiter': 100})
        logger.debug(
            f'{fIdx:04d}/{len(selected_frames):04d} -- Step 2. final loss values: {" | ".join(["{} = {:2.2e}".format(k, np.sum(v.r ** 2)) for k, v in opt_objs.items()])}')


        for k, v in opt_objs.items():
            if k not in perframe_data['stageii_errs']: perframe_data['stageii_errs'][k] = []
            perframe_data['stageii_errs'][k].append(np.sum(v.r ** 2))

        perframe_data['markers_sim'].append(markers_sim.r.copy())
        perframe_data['markers_obs'].append(markers_obs)
        perframe_data['fullpose'].append(opt_model.fullpose.r.copy())
        perframe_data['trans'].append(opt_model.trans.r.copy())



    stageii_data = {k: np.array(v) for k, v in perframe_data.items()}
    stageii_data['betas'] = betas
    return stageii_data


class MoSh:
    """
    The role of the head is to ensure a flexible input/output to various implementations of MoSh
    """

    def __init__(self) -> None:
        super(MoSh, self).__init__()



    def mosh_stagei(self, x):

        # 进行stage i fit
        tm = time.time()
        stagei_data = mosh_stagei_chumpy(stagei_frames=x)
        stagei_elapsed_time = time.time() - tm
        logger.debug(f'finished mosh stagei in {timedelta(seconds=stagei_elapsed_time)}')
        return stagei_data

    def mosh_stageii(self, sequence_data, stagei_data=None):
        if stagei_data is None:
            raise ValueError(f'stagei_fname results could not be found: {self.stagei_fname}. please run stagei first.')


        tm = time.time()
        stageii_data = mosh_stageii_chumpy(sequence_data, stagei_data=stagei_data)
        stageii_elapsed_time = time.time() - tm
        logger.debug(f'finished mosh stageii in {timedelta(seconds=stageii_elapsed_time)}')

        return stageii_data


def worker(rank, total_parts):
    """
    This function should be self-contained; i.e. module imports should all be inside
    :param cfg:
    :return:
    """
    print(f'Proc {rank} start.')
    mp = MoSh()
    test_fp = '/home/lanhai/PycharmProjects/GLRBM-Mocap/data/BMLrub_lmdb_moshpp'
    output_lmdb_path = '/home/lanhai/PycharmProjects/GLRBM-Mocap/data/BMLrub_lmdb_moshpp_results'
    test_dataset = AmassLmdbDataset(test_fp, use_rela_x=False, marker_type='moshpp', device='cuda')
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    os.makedirs(os.path.dirname(output_lmdb_path), exist_ok=True)

    env = lmdb.open(
        output_lmdb_path,
        map_size=10 * 1024 ** 3,  # 预留 50GB，可按需要调整
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False
    )

    with env.begin(write=True) as txn:
        txn.put(b"__info__", pickle.dumps({"description": "Mosh++ processed sequences"}))

    for data_idx, data in enumerate(tqdm(testloader, position=rank, desc=f'Proc {rank}', leave=True)):
        key = f"{data_idx:06d}".encode("ascii")

        if data_idx % total_parts != rank:
            continue

        # 检查是否已存在，避免重复计算
        with env.begin(write=False) as txn:
            if txn.get(key) is not None:
                continue
        
        

        B = data["marker_info"].shape[0]
        L = data["marker_info"].shape[1]
        x = data["marker_info"].contiguous().view(B, L, -1)
        stagei_ind = torch.randint(0, L, (12,))

        stagei_data = mp.mosh_stagei(x[0, stagei_ind, :].cpu().numpy())

        stageii_data = mp.mosh_stageii(x[0, :, :].cpu().numpy(), stagei_data)


        # 二进制序列化
        value = pickle.dumps(stageii_data, protocol=pickle.HIGHEST_PROTOCOL)

        # 写入 LMDB
        with env.begin(write=True) as txn:
            txn.put(key, value)

        # 可选：每隔 N 条提交一次（提升性能）
        if data_idx % 5 == 0:
            env.sync()
        # print(stageii_data)
        # vis_diff_aitviewer(z
        #     "smpl",
        #     gt_full_poses=data["poses"][0],
        #     gt_betas=data["betas"][0],
        #     gt_trans=data["trans"][0],
        #     pred_full_poses=torch.from_numpy(stageii_data["fullpose"]),
        #     pred_betas=torch.from_numpy(stageii_data["betas"]),
        #     pred_trans=torch.from_numpy(stageii_data["trans"])
        # )
    env.close()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn',force=True)
    total_parts = 4
    procs = [
        multiprocessing.Process(target=worker, args=(r, total_parts))
        for r in range(total_parts)
    ]
    [p.start() for p in procs]
    [p.join() for p in procs]
