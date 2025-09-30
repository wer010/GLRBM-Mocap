import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
import trimesh
import pickle
from lbs import (
    lbs, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords, blend_shapes)
VERTEX_IDS = {
    'smplh': {
        'nose':		    332,
        'reye':		    6260,
        'leye':		    2800,
        'rear':		    4071,
        'lear':		    583,
        'rthumb':		6191,
        'rindex':		5782,
        'rmiddle':		5905,
        'rring':		6016,
        'rpinky':		6133,
        'lthumb':		2746,
        'lindex':		2319,
        'lmiddle':		2445,
        'lring':		2556,
        'lpinky':		2673,
        'LBigToe':		3216,
        'LSmallToe':	3226,
        'LHeel':		3387,
        'RBigToe':		6617,
        'RSmallToe':    6624,
        'RHeel':		6787
    },
    'smplx': {
        'nose':		    9120,
        'reye':		    9929,
        'leye':		    9448,
        'rear':		    616,
        'lear':		    6,
        'rthumb':		8079,
        'rindex':		7669,
        'rmiddle':		7794,
        'rring':		7905,
        'rpinky':		8022,
        'lthumb':		5361,
        'lindex':		4933,
        'lmiddle':		5058,
        'lring':		5169,
        'lpinky':		5286,
        'LBigToe':		5770,
        'LSmallToe':    5780,
        'LHeel':		8846,
        'RBigToe':		8463,
        'RSmallToe': 	8474,
        'RHeel':  		8635
    },
    'mano': {
            'thumb':		744,
            'index':		320,
            'middle':		443,
            'ring':		    554,
            'pinky':		671,
        }
}


class Smpl(nn.Module):
    """
    PyTorch implementation of SMPL model with Linear Blend Skinning
    """
    NUM_BODY_JOINTS = 23
    NUM_JOINTS = NUM_BODY_JOINTS

    def __init__(self,
                 model_path,
                 batch_size=1,
                 num_betas: int = 10,
                 device: str = 'cpu'):
        super(Smpl, self).__init__()

        self.batch_size = batch_size
        self.num_betas = num_betas

        self.to(device)
        dd = np.load(model_path, allow_pickle=True)

        # Convert smpl model parameters to torch tensors and register as buffers
        self.faces = dd['f']
        self.register_buffer('faces_tensor', torch.from_numpy(self.faces).long().to(device))
        self.vertices = dd['v_template']
        self.register_buffer('v_template', torch.from_numpy(self.vertices).float().to(device))
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        self.register_buffer('vertex_normals', torch.from_numpy(mesh.vertex_normals.copy()).float().to(device))
        posedirs = np.reshape(dd['posedirs'], [-1, dd['posedirs'].shape[-1]]).T
        self.register_buffer('posedirs', torch.from_numpy(posedirs).float().to(device))
        shapedirs = dd['shapedirs']
        shapedirs = shapedirs[:, :, :num_betas]
        self.register_buffer('shapedirs', torch.from_numpy(shapedirs).float().to(device))
        self.register_buffer('lbs_weights', torch.from_numpy(dd['weights']).float().to(device))
        self.register_buffer('J_regressor', torch.from_numpy(dd['J_regressor']).float().to(device))
        parents = dd['kintree_table'][0]
        parents[0] = -1
        self.register_buffer('parents', torch.from_numpy(parents).long().to(device))

        default_body_pose = torch.zeros([batch_size, self.NUM_BODY_JOINTS * 3])
        self.body_pose = nn.Parameter(default_body_pose, requires_grad=True).to(device)

        vertex_ids = VERTEX_IDS['smplh']
        self.vertex_joint_selector = VertexJointSelector(vertex_ids=vertex_ids, device = device)

        vertex_faces = Smpl.vertex_faces_sorted(mesh)
        self.register_buffer('vertex_faces', torch.from_numpy(vertex_faces).long().to(device))


    def forward(
            self,
            betas: Optional[torch.Tensor] = None,
            body_pose: Optional[torch.Tensor] = None,
            global_orient: Optional[torch.Tensor] = None,
            transl: Optional[torch.Tensor] = None,
            return_verts=True,
            return_full_pose: bool = False,
            pose2rot: bool = True,
            **kwargs
    ):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = body_pose.shape[0]

        if betas.dim()==1:
            betas = betas.expand(batch_size, -1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)

        # joints = self.vertex_joint_selector(vertices, joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = {'vertices':vertices if return_verts else None,
                'global_orient':global_orient,
                'body_pose':body_pose,
                'joints':joints,
                'betas':betas,
                'full_pose':full_pose if return_full_pose else None}

        return output

    @staticmethod
    def vertex_faces_sorted(mesh):
        # 把某点对应的face id按三角形面积从小到大排列

        area = mesh.area_faces[mesh.vertex_faces]

        mask = (mesh.vertex_faces == -1)  # 去除无效项

        area[mask] = np.inf

        vertex_faces_id  = np.take_along_axis(mesh.vertex_faces, np.argsort(area, axis=-1), axis=-1)
        return vertex_faces_id


class Smplx(nn.Module):
    """
    PyTorch implementation of SMPL model with Linear Blend Skinning
    """
    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS

    def __init__(self, 
                 model_path,                 
                 batch_size=1,
                 num_betas: int = 10,
                 num_expression_coeffs: int = 10,
                 num_pca_comps: int = 6,
                 device: str = 'cpu'):
        super(Smplx, self).__init__()
        
        self.batch_size = batch_size
        self.num_betas = num_betas
        self.num_expression_coeffs = num_expression_coeffs
        self.num_pca_comps = num_pca_comps
        self.to(device)
        dd = np.load(model_path, allow_pickle=True)

        # Convert smpl model parameters to torch tensors and register as buffers
        self.faces = dd['f']
        self.register_buffer('faces_tensor', torch.from_numpy(dd['f'].astype(np.int64)).long())
        self.register_buffer('v_template', torch.from_numpy(dd['v_template']).float())
        posedirs = np.reshape(dd['posedirs'], [-1, dd['posedirs'].shape[-1]]).T
        self.register_buffer('posedirs', torch.from_numpy(posedirs).float())
        self.register_buffer('lbs_weights', torch.from_numpy(dd['weights']).float())
        self.register_buffer('J_regressor', torch.from_numpy(dd['J_regressor']).float())
        parents = dd['kintree_table'][0]
        parents[0]=-1
        self.register_buffer('parents', torch.from_numpy(parents).long())

        default_body_pose = torch.zeros([batch_size, self.NUM_BODY_JOINTS * 3])
        self.body_pose= nn.Parameter(default_body_pose, requires_grad=True)

        
        # Convert smplh model parameters (hands)
        hands_componentsl = dd['hands_componentsl'][:num_pca_comps] 
        hands_meanl = dd['hands_meanl']
        hands_componentsr = dd['hands_componentsr'][:num_pca_comps]
        hands_meanr = dd['hands_meanr']
        self.register_buffer('left_hand_components', torch.from_numpy(hands_componentsl).float())
        self.register_buffer('right_hand_components', torch.from_numpy(hands_componentsr).float())
        self.register_buffer('left_hand_mean', torch.from_numpy(hands_meanl).float())
        self.register_buffer('right_hand_mean', torch.from_numpy(hands_meanr).float())
        
        default_lhand_pose = torch.zeros([batch_size, num_pca_comps])
        default_rhand_pose = torch.zeros([batch_size, num_pca_comps])
        self.left_hand_pose = nn.Parameter(default_lhand_pose, requires_grad=True)
        self.right_hand_pose = nn.Parameter(default_rhand_pose, requires_grad=True)


        # Convert smplx model parameters (face)
        lmk_faces_idx = dd['lmk_faces_idx']
        self.register_buffer('lmk_faces_idx',
                             torch.tensor(lmk_faces_idx, dtype=torch.long))
        lmk_bary_coords = dd['lmk_bary_coords']
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_bary_coords, dtype=torch.float32))
        
        default_jaw_pose = torch.zeros([batch_size, 3], dtype=torch.float32)
        default_leye_pose = torch.zeros([batch_size, 3], dtype=torch.float32)
        default_reye_pose = torch.zeros([batch_size, 3], dtype=torch.float32)

        self.jaw_pose = nn.Parameter(default_jaw_pose, requires_grad=True)
        self.leye_pose = nn.Parameter(default_leye_pose, requires_grad=True)
        self.reye_pose = nn.Parameter(default_reye_pose, requires_grad=True)
        
        shapedirs = dd['shapedirs']
        expr_start_idx = 300
        expr_end_idx = 300 + num_expression_coeffs
        expr_dirs = shapedirs[:, :, expr_start_idx:expr_end_idx]
        self.register_buffer('expr_dirs', torch.from_numpy(expr_dirs).float())
        
        shapedirs = shapedirs[:, :, :num_betas]
        self.register_buffer('shapedirs', torch.from_numpy(shapedirs).float())


        default_expression = torch.zeros([batch_size, num_expression_coeffs], dtype=torch.float32)
        self.expression = nn.Parameter(default_expression, requires_grad=True)
        
        global_orient_mean = torch.zeros([3], dtype=torch.float32)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3],
                                     dtype=torch.float32)
        jaw_pose_mean = torch.zeros([3], dtype=torch.float32)
        leye_pose_mean = torch.zeros([3], dtype=torch.float32)
        reye_pose_mean = torch.zeros([3], dtype=torch.float32)
        pose_mean = torch.concatenate([global_orient_mean, body_pose_mean,
                                    jaw_pose_mean,
                                    leye_pose_mean, reye_pose_mean,
                                    torch.zeros_like(self.left_hand_mean),
                                       torch.zeros_like(self.right_hand_mean)],
                                   axis=0)
        pose_mean_tensor = pose_mean.clone().detach().to(torch.float32)
        self.register_buffer('pose_mean', pose_mean_tensor)
        vertex_ids = VERTEX_IDS['smplx']
        self.vertex_joint_selector = VertexJointSelector(vertex_ids=vertex_ids, device = device)

        

            
    def forward(
        self,
        betas: Optional[torch.Tensor] = None,
        global_orient: Optional[torch.Tensor] = None,
        body_pose: Optional[torch.Tensor] = None,
        left_hand_pose: Optional[torch.Tensor] = None,
        right_hand_pose: Optional[torch.Tensor] = None,
        transl: Optional[torch.Tensor] = None,
        expression: Optional[torch.Tensor] = None,
        jaw_pose: Optional[torch.Tensor] = None,
        leye_pose: Optional[torch.Tensor] = None,
        reye_pose: Optional[torch.Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        return_shaped: bool = True,
        **kwargs):

        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        left_hand_pose = (left_hand_pose if left_hand_pose is not None else
                          self.left_hand_pose)
        right_hand_pose = (right_hand_pose if right_hand_pose is not None else
                           self.right_hand_pose)
        jaw_pose = jaw_pose if jaw_pose is not None else self.jaw_pose
        leye_pose = leye_pose if leye_pose is not None else self.leye_pose
        reye_pose = reye_pose if reye_pose is not None else self.reye_pose
        expression = expression if expression is not None else self.expression

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        left_hand_pose = torch.einsum(
            'bi,ij->bj', [left_hand_pose, self.left_hand_components])
        right_hand_pose = torch.einsum(
            'bi,ij->bj', [right_hand_pose, self.right_hand_components])

        full_pose = torch.cat([global_orient.reshape(-1, 1, 3),
                               body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3),
                               jaw_pose.reshape(-1, 1, 3),
                               leye_pose.reshape(-1, 1, 3),
                               reye_pose.reshape(-1, 1, 3),
                               left_hand_pose.reshape(-1, 15, 3),
                               right_hand_pose.reshape(-1, 15, 3)],
                              dim=1).reshape(-1, 165)

        # Add the mean pose of the model. Does not affect the body, only the
        # hands when flat_hand_mean == False
        full_pose += self.pose_mean

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])
        # Concatenate the shape and expression coefficients
        scale = int(batch_size / betas.shape[0])
        if scale > 1:
            betas = betas.expand(scale, -1)
            expression = expression.expand(scale, -1)
        shape_components = torch.cat([betas, expression], dim=-1)

        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)

        vertices, joints = lbs(shape_components, full_pose, self.v_template,
                               shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot,
                               )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            self.batch_size, 1, 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)
        # Map the joints to the current dataset


        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        v_shaped = None
        if return_shaped:
            v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        else:
            v_shaped = torch.Tensor(0)
        output = {
            'vertices': vertices if return_verts else None,
            'joints': joints,
            'betas': betas,
            'expression': expression,
            'global_orient': global_orient,
            'transl': transl,
            'body_pose': body_pose,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
            'jaw_pose': jaw_pose,
            'v_shaped': v_shaped,
            'full_pose': full_pose if return_full_pose else None
        }
        return output

class VertexJointSelector(nn.Module):

    def __init__(self, vertex_ids=None,
                 use_hands=True,
                 use_feet_keypoints=True, **kwargs):
        super(VertexJointSelector, self).__init__()

        extra_joints_idxs = []

        face_keyp_idxs = np.array([
            vertex_ids['nose'],
            vertex_ids['reye'],
            vertex_ids['leye'],
            vertex_ids['rear'],
            vertex_ids['lear']], dtype=np.int64)

        extra_joints_idxs = np.concatenate([extra_joints_idxs,
                                            face_keyp_idxs])

        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'],
                                       vertex_ids['LSmallToe'],
                                       vertex_ids['LHeel'],
                                       vertex_ids['RBigToe'],
                                       vertex_ids['RSmallToe'],
                                       vertex_ids['RHeel']], dtype=np.int32)

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, feet_keyp_idxs])

        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, tips_idxs])

        self.register_buffer('extra_joints_idxs',
                             torch.from_numpy(extra_joints_idxs).long().to(kwargs['device']))

    def forward(self, vertices, joints):
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs.to(torch.long)) #The '.to(torch.long)'.
                                                                                            # added to make the trace work in c++,
                                                                                            # otherwise you get a runtime error in c++:
                                                                                            # 'index_select(): Expected dtype int32 or int64 for index'
        joints = torch.cat([joints, extra_joints], dim=1)

        return joints

