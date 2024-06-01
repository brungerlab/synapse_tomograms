import numpy as np
from scipy.spatial import Delaunay, Voronoi, delaunay_plot_2d, voronoi_plot_2d
import matplotlib.pyplot as plt
import alphashape
import pandas as pd
import glob
import math

def csv_to_np(filename):
    data_points = pd.read_csv(filename, header=1).to_numpy()[:,:3].astype(np.float64)

    return data_points

def Bounding_vol(vertices,alpha=0.005):
    #Create a polygon object that defines the volume boundaries e.g. the synaptic cleft
    vol_points = pd.read_csv(vertices, header=1).to_numpy()
    volume = alphashape.alphashape(vol_points, alpha=alpha)

    return volume

def synth_cluster_centers(data_points, min_d, volume, iterations):
    '''
    :param data_points: original coordinates of the cluster centers to be randomized
    :param min_d: minimum distance between cluster coordinates
    :param volume: boundary volume to be populated. The same number of points as the data will be placed within this vol
    :param iterations: number of synthetic datasets to generate

    :return numpy array [number of clusters x coordinates x number of sim iterations]
    '''

    # define the boundaries of the area to be populated
    x = volume.vertices[:, 0]
    y = volume.vertices[:, 1]
    z = volume.vertices[:, 2]
    xrange = (max(x) - min(x))
    yrange = (max(y) - min(y))
    zrange = (max(z) - min(z))
    region = [xrange, yrange, zrange]

    sims = np.zeros((len(data_points),3,iterations))

    for it in np.arange(0,iterations):
        keep_x = xrange * np.random.rand() + np.min(x)
        keep_y = yrange * np.random.rand() + np.min(y)
        keep_z = zrange  * np.random.rand() + np.min(z)
        point = np.hstack((keep_x,keep_y,keep_z)).reshape(1,3)
        a = 0
        while (a < 1):
            if volume.contains(point):  # and (cleft.contains(keep)==False):
                a += 1
                keep = point.copy()
            else:
                keep_x = xrange * np.random.rand() + np.min(x)
                keep_y = yrange * np.random.rand() + np.min(y)
                keep_z = zrange * np.random.rand() + np.min(z)
                point = np.hstack((keep_x, keep_y, keep_z)).reshape(1,3)
        while len(keep) < len(data_points):
            keep_x = xrange * np.random.rand() + np.min(x)
            keep_y = yrange * np.random.rand() + np.min(y)
            keep_z = zrange * np.random.rand() + np.min(z)
            point = np.hstack((keep_x, keep_y, keep_z)).reshape(1,3)
            if volume.contains(point):  # and (cleft.contains(point)==False):
                dst = np.linalg.norm(point - keep, axis=1)
                too_close = np.where(dst < min_d)
                if len(too_close[0]) == 0:
                    keep = np.concatenate((keep, point))
        sims[:,:,it] = keep

    return sims

def np_cluster_NND(Same_points,Opp_points,sort=False):
    # empty arrays for the pairwise distance measurements each direction
    Dist_Mat = np.zeros((len(Same_points), len(Opp_points)))
    #euclidean distance between points of opposite type
    for i in np.arange(0, len(Same_points)):
        Dist_Mat[i, :] = np.linalg.norm(Same_points[i] - Opp_points,axis=1)

    if sort==True:
        Dist_Mat = np.sort(Dist_Mat,axis=1)

    return Dist_Mat

def Biv_G(Dist_Mat,max_rad=200,SelfNND=False):
    '''

    :param max_rad: maximum distance to consider
    :param Dist_Mat: euclidean distance matrix between point type I and type J
    :return:
    '''
    if SelfNND==True:
        idx = 1
    else:
        idx = 0
    NI,NJ = np.shape(Dist_Mat)
    rad_bins = np.arange(0,max_rad,1)
    GIJ = np.zeros((len(rad_bins), 1))
    GJI = np.zeros((len(rad_bins), 1))
    DMIJ = np.sort(Dist_Mat,axis=1)
    DMJI = np.sort(Dist_Mat.T, axis=1)
    for k in np.arange(0,len(GIJ)):
        k_idx = np.argwhere(DMIJ[:,idx]<(rad_bins[k]))
        GIJ[k]=len(k_idx)/NI
    for j in np.arange(0,len(GJI)):
        j_idx = np.argwhere(DMJI[:,idx]<(rad_bins[j]))
        GJI[j]=len(j_idx)/NJ
    GIJ = GIJ.reshape((max_rad,))
    GJI = GJI.reshape((max_rad,))

    return GIJ,GJI

def sims_Biv_G(I_points,J_points,volume,iterations,rad_max=200,min_d=30,SelfNND=False):

    synth = synth_cluster_centers(I_points,min_d, volume, iterations)
    G_sims_ItJ = np.zeros((rad_max,iterations))
    G_sims_JtI = np.zeros((rad_max, iterations))

    if SelfNND==True:
        for i in np.arange(0, iterations):
            sItJ_dist_mat = np_cluster_NND(synth[:, :, i], synth[:, :, i])
            G_sims_ItJ[:, i], G_sims_JtI[:, i] = Biv_G(sItJ_dist_mat, max_rad=rad_max, SelfNND=SelfNND)
    else:
        for i in np.arange(0, iterations):
            sItJ_dist_mat = np_cluster_NND(synth[:, :, i], J_points)
            G_sims_ItJ[:, i], G_sims_JtI[:, i] = Biv_G(sItJ_dist_mat, max_rad=rad_max, SelfNND=SelfNND)

    return G_sims_ItJ,G_sims_JtI

def Loop_Biv_G(I_files,J_files,vol_files,max_rad=200,iterations=99,min_d=30,SelfNND=False):
    '''
    Loop through files to calculate and compile the bivariate NND (G function) vs control simulations. Make sure the two
    lists are sorted such that the matched files are in the same order.
    :param I_files:
    :param J_files:
    :param vol_files:
    :param max_rad:
    :param iterations:
    :return:
    '''

    G_ItJ = np.zeros((max_rad,(iterations+1),len(I_files)))
    G_JtI = G_ItJ.copy()

    for i in np.arange(0,len(I_files)):
        print(str(I_files[i]))
        I_points = csv_to_np(I_files[i])
        J_points = csv_to_np(J_files[i])
        volume = Bounding_vol(vol_files[i])

        Dist_Mat = np_cluster_NND(I_points,J_points)
        G_IJ,G_JI = Biv_G(Dist_Mat,max_rad=max_rad,SelfNND=SelfNND)
        G_IJ = np.reshape(G_IJ,(max_rad,1))
        G_JI = np.reshape(G_JI, (max_rad, 1))
        Gsim_IJ, Gsim_JI = sims_Biv_G(I_points,J_points,volume,iterations,rad_max=max_rad,min_d=min_d,SelfNND=SelfNND)
        G_IJ = np.hstack((G_IJ,Gsim_IJ))
        G_JI = np.hstack((G_JI,Gsim_JI))

        G_ItJ[:,:,i] = G_IJ
        G_JtI[:,:,i] = G_JI

    return G_ItJ, G_JtI

def match_pairs(DF1,DF2):
    '''

    :param DF1: pandas dataframe of one cluster type. Needs to have OppCluster_NND in reference to the cluster type in DF2
    :param DF2: pandas dataframe of second cluster type. Needs to have OppCluster_NND in reference to the cluster type in DF1
    :return: DFx_pair_idx - indices of clusters that are paired. unpaired can be indexed using ==False.
    '''

    DF1_pair_idx = DF1['OppCluster_NND'].isin(DF2['OppCluster_NND'])
    DF2_pair_idx = DF2['OppCluster_NND'].isin(DF1['OppCluster_NND'])

    return DF1_pair_idx,DF2_pair_idx

def G_fun_plot(G_fun,CI=0.95,color='red',error=True):
    data_mean = np.mean(G_fun[:, 0, :], axis=1)
    sem = np.std(G_fun[:, 0, :], axis=1)/np.sqrt(np.shape(G_fun)[2])
    mean = np.mean(np.mean(G_fun, axis=2), axis=1)
    sim_srt = np.sort(np.mean(G_fun[:, 1:, :], axis=2),axis=1)
    lower = np.rint(((1-CI) * np.shape(sim_srt)[1])/2).astype(int)
    upper = (np.shape(sim_srt)[1] - lower - 1).astype(int)

    cm = 1/2.54 #cm to inch
    fig, (ax1, ax2) = plt.subplots(ncols=2,sharex=True,figsize=(11*cm,5*cm))
    rng = np.arange(0,np.shape(G_fun)[0],1)
    ax1.plot(rng, mean, color='black')
    ax1.fill_between(rng, sim_srt[:,lower],sim_srt[:,upper],color='black',alpha=0.2)
    ax1.plot(rng, data_mean, color=color)
    if error==True:
        ax1.fill_between(rng, data_mean - sem, data_mean + sem, color=color, alpha=0.2)

    bins = np.arange(0,200,7.5)
    dig = np.digitize(rng[:-1],bins)
    data_diff = np.diff(data_mean)
    sim_diff = np.diff(mean)
    data_his = [data_diff[dig == i].sum() for i in np.arange(1,len(bins))]
    sim_his = [sim_diff[dig == i].sum() for i in np.arange(1,len(bins))]
    ax2.bar(bins[1:] - (7.5 / 2), np.array(sim_his), width=7.5, color='black', alpha=0.5)
    ax2.bar(bins[1:]-(7.5/2),np.array(data_his),width=7.5,color=color,alpha=0.5)

    fig.show()

    return fig, ax1, ax2, data_his, sim_his

def MAD_test(G_fun):
    '''
    Maximum Absolute Difference (MAD) test. This is an envelope test using the maximum absolute difference between the
    cumulative histogram of each trial (data or sim) vs the mean (data + sims). P value is calculated as the number of
    simulation trials with MAD greater than that of the true data, divided by total trials.
    :param G_fun: 3D array of all data [bins,trials (data + sims) ,tomograms]. Real data should be at [:,0,:]
    :return: P-value by MAD test
    '''
    data_mean = np.mean(G_fun[:,0,:],axis=1)
    sims = np.mean(G_fun[:,1:,:],axis=2)
    mean = np.mean(np.mean(G_fun,axis=2),axis=1)
    Tdata = np.max(np.abs(data_mean.flatten()-mean.flatten()))
    tmp = []
    for i in np.arange(0,np.shape(sims)[1]):
        tmp.append(np.max(np.abs(sims[:,i].flatten()-mean.flatten())))
    Tsim = np.array(tmp)
    ct = len(np.where(Tsim>Tdata)[0])
    p = (1+ct)/np.shape(G_fun)[1]

    return p


def array_save(np_array,filename):
    with open((str(filename)+'.txt'), 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(np_array.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in np_array:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New tomo\n')

    return
