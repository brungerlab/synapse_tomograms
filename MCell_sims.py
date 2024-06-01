import numpy as np
import pandas as pd
import itertools
import glob
import matplotlib.pyplot as plt

clust_file = []
hom_file = []
for i in tomos:
    clust_file.append(glob.glob((i + '_files/mcell/Clustered/Vesicle_*_output_data/react_data/seed_*/O_World.dat')))
    hom_file.append(glob.glob((i + '_files/mcell/Uniform/Vesicle_*_output_data/react_data/seed_*/O_World.dat')))

clust_file = list(itertools.chain.from_iterable(clust_file))
hom_file = list(itertools.chain.from_iterable(hom_file))
clust_file.sort()
hom_file.sort()

clust_data = np.zeros((50,10000))
hom_data = np.zeros((50,10000))

for i in range(0,len(clust_file)):
    clust_data[i,:] = np.genfromtxt(clust_file[i],delimiter=' ')[1:,1].astype(int)
    hom_data[i, :] = np.genfromtxt(hom_file[i], delimiter=' ')[1:, 1].astype(int)

clust = clust_data.reshape(-1,50,clust_data.shape[-1]).mean(1)
hom = hom_data.reshape(-1,50,hom_data.shape[-1]).mean(1)
data = []
p=0
for t in range(0,len(tomos)):
    ves_num = len(glob.glob(tomos[t]+'_files/mcell/Clustered/Vesicle_*_output_data'))
    for v in range(0,ves_num):
        row = [tomos[t],v,np.max(clust[p,:]),np.max(hom[p,:])]
        data.append(row)
        p += 1

DF = pd.DataFrame(data, columns=['Tomo', 'Vesicle', 'Clust_Max', 'Hom_Max'])

clust_mean=[]
clust_std=[]
hom_mean=[]
hom_std=[]

for tomo in tomos:
    clust_mean.append(np.mean(DF.loc[DF['Tomo']==tomo]['Clust_Max']))
    clust_std.append(np.std(DF.loc[DF['Tomo'] == tomo]['Clust_Max']))
    hom_mean.append(np.mean(DF.loc[DF['Tomo'] == tomo]['Hom_Max']))
    hom_std.append(np.std(DF.loc[DF['Tomo'] == tomo]['Hom_Max']))

clust_mean = np.array(clust_mean)
clust_std = np.array(clust_std)
hom_mean = np.array(hom_mean)
hom_std = np.array(hom_std)
clust_cv = clust_std/clust_mean
hom_cv = hom_std/hom_mean
