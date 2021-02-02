import gzip
import torch.utils.data as data
import lmdb
import msgpack_numpy
import os
import os.path as osp
import random
import numpy as np
import torch
class THERMO(data.Dataset):
    def __init__(self,root):
        self.root = root
        self._cache = os.path.join(self.root, "thermo_cache.npy")
        if not os.path.isfile(self._cache):
            sumarray = np.array([])
            for file in os.listdir(self.root):
                if file=="thermo_cache":
                    continue
                print("processing",file)
                nparray = np.array([])
                with gzip.open(osp.join(self.root,file),'rb') as f:
                    for n,line in enumerate(f):
                        if 23 <= n <= 320022:
                            nparray = np.append(nparray,float(line[:-1]))
                sumarray = np.append(sumarray,nparray)
            sumarray = sumarray.reshape(320000,-1)
            np.save(self._cache,sumarray)


        self._lmdb_file = np.load(self._cache)
    def __getitem__(self, index):
        if self._lmdb_file is None:
            self._lmdb_file = np.load(self._cache)
        slice = self._lmdb_file[index,:]
        input, output = np.append(slice[:17],slice[-1]),slice[17:-1]
        input = torch.tensor(input,dtype=torch.float32)
        output = torch.tensor(output,dtype=torch.float32)
        return input.unsqueeze(-1), output
    def __len__(self):
        return 320000
if __name__ == '__main__':
    dataset = THERMO('data/0.022000625/')
    input,output = random.choice(dataset)
    print(input.shape,output.shape)
