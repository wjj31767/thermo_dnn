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
        self.dict = {"CH":0,
                    "CH2":1,
                    "CH2O":2,
                    "CH3":3,
                    "CH4":4,
                    "CO":5,
                    "CO2":6,
                    "H":7,
                    "H2":8,
                    "H2O":9,
                    "H2O2":10,
                    "HCO":11,
                    "HO2":12,
                    "N2":13,
                    "O":14,
                    "O2":15,
                    "OH":16,
                    "T":17,
                    "RR.CH":18,
                    "RR.CH2":19,
                    "RR.CH2O":20,
                    "RR.CH3":21,
                    "RR.CH4":22,
                    "RR.CO":23,
                    "RR.CO2":24,
                    "RR.H":25,
                    "RR.H2":26,
                    "RR.H2O":27,
                    "RR.H2O2":28,
                    "RR.HCO":29,
                    "RR.HO2":30,
                    "RR.O":31,
                    "RR.O2":32,
                    "RR.OH":33}
        if not os.path.isfile(self._cache):
            sumarray = []
            for sublist in sorted(os.listdir(self.root)):
                subarray = [np.zeros(4000).reshape(-1, 1) for _ in range(34)]
                # if sublist=="thermo_cache.npy":
                if sublist not in ['0.5']:
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
        if self.use_norm:
            self.summean = self._lmdb_file.mean(axis=0, keepdims=True)
            self.sumstd = self._lmdb_file.std(axis=0, keepdims=True)
            self.mask = (self.sumstd!=0.).squeeze()
            self._lmdb_file[:,self.mask] = (self._lmdb_file[:,self.mask] - self.summean[:,self.mask]) / self.sumstd[:,self.mask]

    def __getitem__(self, index):
        if self._lmdb_file is None:
            self._lmdb_file = np.load(self._cache)
            if self.use_norm:
                self.summean = self._lmdb_file.mean(axis=0, keepdims=True)
                self.sumstd = self._lmdb_file.std(axis=0, keepdims=True)
                mask = (self.sumstd != 0.).squeeze()
                self._lmdb_file[:, mask] = (self._lmdb_file[:, mask] - self.summean[:, mask]) / self.sumstd[:, mask]

        slice = self._lmdb_file[index,:]
        input, output = slice[:18],slice[18:]
        input = torch.tensor(input,dtype=torch.float32)
        output = torch.tensor(output,dtype=torch.float32)
        return input.unsqueeze(-1), output
    def __len__(self):
        return self._lmdb_file.shape[0]
if __name__ == '__main__':
    dataset = THERMO('data/',True)
    input,output = random.choice(dataset)
    print(input.shape,output,input)
    print(len(dataset))
