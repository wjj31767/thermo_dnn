import gzip
import torch.utils.data as data
import os
import os.path as osp
import random
import numpy as np
import torch
class THERMO(data.Dataset):
    def __init__(self,root,use_norm):
        self.root = root
        self.use_norm=use_norm
        self._cache = os.path.join(self.root, "thermo_cache.npy")
        if not os.path.isfile(self._cache):
            sumarray = []
            for file in sorted(os.listdir(self.root)):
                if file=="thermo_cache.npy":
                    continue
                print("processing",file)
                nparray = np.array([])
                with gzip.open(osp.join(self.root,file),'rb') as f:
                    for n,line in enumerate(f):
                        if 23 <= n <= 320022:
                            nparray = np.append(nparray,float(line[:-1]))
                sumarray.append(nparray.reshape(-1,1))
            sumarray = np.hstack(sumarray)
            np.save(self._cache,sumarray)



        self._lmdb_file = np.load(self._cache)
        if self.use_norm:
            self.summean = self._lmdb_file.mean(axis=0, keepdims=True)
            self.sumstd = self._lmdb_file.std(axis=0, keepdims=True)
            self._lmdb_file = (self._lmdb_file - self.summean) / self.sumstd
    def __getitem__(self, index):
        if self._lmdb_file is None:
            self._lmdb_file = np.load(self._cache)
            if self.use_norm:
                self.summean = self._lmdb_file.mean(axis=0, keepdims=True)
                self.sumstd = self._lmdb_file.std(axis=0, keepdims=True)
                self._lmdb_file = (self._lmdb_file - self.summean) / self.sumstd
        slice = self._lmdb_file[index,:]
        input, output = np.append(slice[:17],slice[-1]),slice[17:-1]
        input = torch.tensor(input,dtype=torch.float32)
        output = torch.tensor(output,dtype=torch.float32)
        return input.unsqueeze(-1), output
    def __len__(self):
        return 320000
if __name__ == '__main__':
    dataset = THERMO('data/0.022000625/',False)
    input,output = random.choice(dataset)
    print(input.shape,output)
