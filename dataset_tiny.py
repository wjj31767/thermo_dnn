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
        self.dict = {
                    "CH4":0,
                    "CO2":1,
                    "H2O":2,
                    "N2":3,
                    "O2":4,
                    "T":5,
                    "RR.CH4":6,
                    "RR.CO2":7,
                    "RR.H2O":8,
                    "RR.O2":9,
        }
        if not os.path.isfile(self._cache):
            sumarray = []
            for sublist in sorted(os.listdir(self.root)):
                subarray = [np.zeros(4000).reshape(-1, 1) for _ in range(10)]
                if sublist=="thermo_cache.npy":
                # if sublist not in ['0.5','0.55']:
                    continue
                print("processing",sublist)
                for file in os.listdir(os.path.join(self.root,sublist)):
                    nparray = np.array([])
                    with open(osp.join(self.root,sublist,file),'rb') as f:
                        for n,line in enumerate(f):
                            if 23 <= n <= 4022:
                                nparray = np.append(nparray,float(line[:-1]))
                    subarray[self.dict[file]] = nparray.reshape(-1,1)
                subarray = np.hstack(subarray)
                sumarray.append(subarray)
            sumarray = np.vstack(sumarray)
            np.save(self._cache,sumarray)



        self._lmdb_file = np.load(self._cache)
        self._lmdb_file = np.unique(self._lmdb_file,axis=0)
        print(self._lmdb_file.shape)
        if self.use_norm=='stand':
            self.summean = self._lmdb_file.mean(axis=0, keepdims=True)
            self.sumstd = self._lmdb_file.std(axis=0, keepdims=True)
            self.mask = (self.sumstd!=0.).squeeze()
            self._lmdb_file[:,self.mask] = (self._lmdb_file[:,self.mask] - self.summean[:,self.mask]) / self.sumstd[:,self.mask]

        elif self.use_norm == 'rescale':
            self.summin = self._lmdb_file.min(axis=0, keepdims=True)
            self.summax = self._lmdb_file.max(axis=0, keepdims=True)
            self.mask = ((self.summin - self.summax) != 0.).squeeze()
            self._lmdb_file[:, self.mask] = (self._lmdb_file[:, self.mask] - self.summin[:, self.mask]) / (
                        self.summax[:, self.mask] - self.summin[:, self.mask])
    def __getitem__(self, index):
        if self._lmdb_file is None:
            self._lmdb_file = np.load(self._cache)
            if self.use_norm:
                self.summean = self._lmdb_file.mean(axis=0, keepdims=True)
                self.sumstd = self._lmdb_file.std(axis=0, keepdims=True)
                mask = (self.sumstd != 0.).squeeze()
                self._lmdb_file[:, mask] = (self._lmdb_file[:, mask] - self.summean[:, mask]) / self.sumstd[:, mask]

        slice = self._lmdb_file[index,:]
        input, output = slice[:6],slice[6:]
        input = torch.tensor(input,dtype=torch.float32)
        output = torch.tensor(output,dtype=torch.float32)
        return input.unsqueeze(-1), output
    def __len__(self):
        return self._lmdb_file.shape[0]
if __name__ == '__main__':
    dataset = THERMO('data/','rescale')
    input,output = random.choice(dataset)
    print(input.shape,output,input)
    print(len(dataset))
