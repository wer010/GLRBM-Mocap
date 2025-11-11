from data import AmassLmdbDataset
from torch.utils.data import DataLoader
from utils import visualize_aitviewer

test_fp = 'data/BMLrub_lmdb_rbm'

test_dataset = AmassLmdbDataset(test_fp, use_rela_x=False, marker_type='rbm', device='cuda')
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for data_idx, data in enumerate(testloader):
    B = data["marker_info"].shape[0]
    L = data["marker_info"].shape[1]

    x_pos = data["marker_info"][..., :3]
    x_ori = data["marker_info"][..., 3:]
    x_pos = x_pos.cpu().numpy()
    x_ori = x_ori.cpu().numpy()
    visualize_aitviewer('smpl', data["poses"][0], data["betas"][0], data["trans"][0], rbs={'pos': x_pos[0], 'ori': x_ori[0]})
