import lmdb
from data import AmassLmdbDataset
from tqdm import tqdm

data_path = 'data/CMU_lmdb_moshpp'

# dataset = AmassLmdbDataset(data_path, use_rela_x=False, marker_type='rbm', device='cuda')

# print(dataset.__len__())

env = lmdb.open(data_path, readonly=True)
with env.begin(write=False) as txn:
    ks  = [key for key, value in txn.cursor()]
    print(len(ks))

# l_list = []
# for i in tqdm(range(len(dataset))):
#     data = dataset[i]
#     l_list.append(data['marker_info'].shape[0])
# print(min(l_list), max(l_list))

