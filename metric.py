"""
Copyright (c) Facebook, Inc. and its affiliates, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
"""
import numpy as np
import quaternion
import torch

from tabulate import tabulate
from geo_utils import rigid_landmark_transform_batch, axis_angle_distance
from pytorch3d.transforms import axis_angle_to_matrix
def _procrustes(X, Y, compute_optimal_scale=True):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420
    Args
      X: array NxM of targets, with N number of points and M point dimensionality
      Y: array NxM of inputs
      compute_optimal_scale: whether we compute optimal scale or force it to be 1
    Returns:
      d: squared error after transformation
      Z: transformed Y
      T: computed rotation
      b: scaling
      c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()
    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY
    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)
    traceTA = s.sum()
    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX
    c = muX - b * np.dot(muY, T)
    return d, Z, T, b, c


class MetricsEngine(object):
    """Helper class to compute metrics over a dataset."""

    def __init__(self):
        """
        :param smpl_model: The SMPL model.
        """
        self.eucl_dists = []  # list of Euclidean distance to ground-truth for each sample and each joint
        self.eucl_dists_pa = []  # Same as self.eucl_dists but Procrustes-aligned
        self.angle_diffs = []  # list of angular difference for each sample

        # List of joints to consider for either metric.
        self.eucl_eval_joints = ['root', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle',
                                 'spine3', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
                                 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
        self.angle_eval_joints = ['l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'spine3',
                                  'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
                                  'l_elbow', 'r_elbow']

    def reset(self):
        """Reset all computations."""
        self.eucl_dists = []
        self.eucl_dists_pa = []
        self.angle_diffs = []

    def _compute_eucl_dist(self, kp3d, kp3d_hat):
        """
        Compute and store the Euclidean distance.
        :param kp3d: A tensor of shape (N, F, J, 3).
        :param kp3d_hat: A tensor of shape (N, F, J, 3).
        :return eucl_dist, eucl_dist_pa (N, F, J) with rigid alignment
        """
        kp3d_shape = kp3d.shape
        kp3d = kp3d.view(-1, *kp3d_shape[-2:])
        kp3d_hat = kp3d_hat.view(-1, *kp3d_shape[-2:])

        diff = kp3d - kp3d_hat
        eucl_dist = torch.norm(diff, dim=-1)


        # Align estimate with reference.

        AA_pred, o_pred = rigid_landmark_transform_batch(kp3d, kp3d_hat)
        R_pred = axis_angle_to_matrix(AA_pred)
        diff_pa = torch.matmul(kp3d, R_pred.transpose(2,1)) + o_pred[:,None,:] - kp3d_hat
        eucl_dist_pa = torch.norm(diff_pa, dim=-1)

        return eucl_dist.reshape(*kp3d_shape[:2],-1), eucl_dist_pa.reshape(*kp3d_shape[:2],-1)


    def _compute_angular_dist(self, pose, pose_hat,):
        pose_shape = pose.shape[:-1]

        angle_diff = axis_angle_distance(pose.reshape(*pose_shape, -1,3), pose_hat.reshape(*pose_shape, -1,3))
        angle_diff = torch.rad2deg(angle_diff)
        return angle_diff

    def compute(self, output, ground_truth):

        eucl_dists, eucl_dists_pa = self._compute_eucl_dist(output['joints'], ground_truth['joints'])

        angle_diffs = self._compute_angular_dist(output['poses'], ground_truth['poses'])

        betas_diffs = torch.norm(output['betas']- ground_truth['betas'], p=1)/output['betas'].shape[-1]

        eucl_mean_per_joint = torch.mean(eucl_dists.view(-1, output['joints'].shape[-2]), dim=0)
        eucl_mean_all = torch.mean(eucl_mean_per_joint)
        eucl_std_all = torch.std(eucl_dists)
        eucl_mean_pa_per_joint = torch.mean(eucl_dists_pa.view(-1, output['joints'].shape[-2]), dim=0)
        eucl_mean_pa_all = torch.mean(eucl_mean_pa_per_joint)
        eucl_std_pa_all = torch.std(eucl_dists_pa)

        angle_mean_per_joint = torch.mean(angle_diffs.view(-1, angle_diffs.shape[-1]), dim=0)
        angle_mean_all = torch.mean(angle_mean_per_joint)
        angle_std_all = torch.std(angle_diffs)

        metrics = {'Betas error': betas_diffs.item(),
                   # 'MPJPE raw': eucl_dists,
                   'MPJPE [mm]': eucl_mean_all.item() * 1000.0,
                   'MPJPE STD': eucl_std_all.item() * 1000.0,
                   'PA-MPJPE [mm]': eucl_mean_pa_all.item() * 1000.0,
                   'PA-MPJPE STD': eucl_std_pa_all.item() * 1000.0,
                   'MPJAE [deg]': angle_mean_all.item(),
                   'MPJAE STD': angle_std_all.item()}
        return metrics

    @staticmethod
    def to_pretty_string(metrics, model_name):
        """Print the metrics onto the console, but pretty."""
        headers, values = [], []
        for k in metrics:
            if k=='MPJPE raw':
                continue
            headers.append(k)
            values.append(metrics[k])
        return tabulate([[model_name] + values], headers=['Model'] + headers)




    def get_metrics(self, eucl_idxs_select=True, angle_idxs_select=True):
        """
        Compute the aggregated metrics that we want to report.
        :return: The computed metrics in a dictionary.
        """
        # Mean and median euclidean distance over all batches and joints.
        if len(self.eucl_dists) > 0:
            eucl_dists = np.concatenate(self.eucl_dists, axis=0)
            eucl_dists_pa = np.concatenate(self.eucl_dists_pa, axis=0)
            eucl_idxs = self.eucl_idxs if eucl_idxs_select else list(range(eucl_dists.shape[1]))

            eucl_mean_per_joint = np.mean(eucl_dists, axis=0)[eucl_idxs]
            eucl_mean_all = np.mean(eucl_mean_per_joint)
            eucl_std_all = np.std(eucl_dists[:, eucl_idxs])
            eucl_mean_pa_per_joint = np.mean(eucl_dists_pa, axis=0)[eucl_idxs]
            eucl_mean_pa_all = np.mean(eucl_mean_pa_per_joint)
            eucl_std_pa_all = np.std(eucl_dists_pa[:, eucl_idxs])
        else:
            eucl_mean_all = 0.0
            eucl_std_all = 0.0
            eucl_mean_pa_all = 0.0
            eucl_std_pa_all = 0.0

        # Mean and median angular difference.
        if len(self.angle_diffs) > 0:
            angle_diffs = np.concatenate(self.angle_diffs, axis=0)
            angle_idxs = self.angle_idxs if angle_idxs_select else list(range(angle_diffs.shape[1]))

            angle_mean_per_joint = np.mean(angle_diffs, axis=0)[angle_idxs]
            angle_mean_all = np.mean(angle_mean_per_joint)
            angle_std_all = np.std(angle_diffs[:, angle_idxs])
        else:
            angle_mean_all = 0.0
            angle_std_all = 0.0

        metrics = {'MPJPE [mm]': eucl_mean_all * 1000.0,
                   'MPJPE STD': eucl_std_all * 1000.0,
                   'PA-MPJPE [mm]': eucl_mean_pa_all * 1000.0,
                   'PA-MPJPE STD': eucl_std_pa_all * 1000.0,
                   'MPJAE [deg]': angle_mean_all,
                   'MPJAE STD': angle_std_all}
        return metrics


    @staticmethod
    def to_tensorboard_log(metrics, writer, global_step, prefix=''):
        """Write metrics to tensorboard."""
        writer.add_scalar('metrics/{}/mje mean'.format(prefix), metrics['MPJPE [mm]'], global_step)
        writer.add_scalar('metrics/{}/mje pa mean'.format(prefix), metrics['PA-MPJPE [mm]'], global_step)
        writer.add_scalar('metrics/{}/mae mean'.format(prefix), metrics['MPJAE [deg]'], global_step)
