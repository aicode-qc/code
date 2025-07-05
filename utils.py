import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import random
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
import pickle
from SFM import *

def get_distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)

def get_norm(p):
    return math.sqrt(p[0]**2+ p[1]**2)

def get_cosine(p1,p2,p3):
    '''
    use three pos to get the cosine between two vector
    '''
    p4=np.zeros(2)
    p4[0]=p2[0]-p1[0]
    p4[1]=p2[1]-p1[1]
    m1=math.sqrt(p4[0]**2+ p4[1]**2)
    m2=math.sqrt(p3[0]**2+ p3[1]**2)
    m=p4[0]*p3[0]+p4[1]*p3[1]
    if m2==0:
        return 0
    return m/(m1*m2)

def anorm(p1,p2,p3,p4):   
    cosine_ij=get_cosine(p1,p2,p3)
    vi_norm=get_norm(p3)
    cosine_ji=get_cosine(p2,p1,p4)
    vj_norm=get_norm(p4)
    dis=get_distance(p1,p2)
    if dis==0:
        return 0
    norm=(vi_norm*cosine_ij+vj_norm*cosine_ji)/dis
    return norm 

def anorm1(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)      

def VIG(seq_,seq_rel,norm_lap_matr = True,qz=0.1):
    seq_ = seq_.squeeze()
    seq_rel =seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]
    V = np.zeros((seq_len,max_nodes,2))     
    fssa_weight0 = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):       
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            fssa_weight0[s,h,h] = qz           
            for k in range(h+1,len(step_)):     
                l2_norm = anorm(step_[h],step_[k],step_rel[h],step_rel[k])  
                l2_norm2 = anorm(step_[k],step_[h],step_rel[k],step_rel[h])
                fssa_weight0[s,h,k] = l2_norm
                fssa_weight0[s,k,h] = l2_norm2
        if norm_lap_matr: 
            for i in range(len(step_)):
                for j in range(len(step_)):
                    if fssa_weight0[s,i,j] >0 :
                        if i!=j :
                            fssa_weight0[s,i,i] += qz-fssa_weight0[s,i,j]
                    elif i!=j:
                        fssa_weight0[s,i,j] = 0
            degree = np.array(fssa_weight0[s].sum(1))
            d_hat = np.diag(np.power(degree, -0.5).flatten())
            fssa_weight0[s] = d_hat.dot(fssa_weight0[s]).dot(d_hat)
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(fssa_weight0).type(torch.float)

def PIG(seq_,seq_rel,norm_lap_matr = True, qz=0.1) :
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]
    V1 = np.zeros((seq_len,max_nodes,2))
    fssa_weight1 = np.zeros((seq_len,max_nodes,max_nodes))
    for h in range(max_nodes):
        traj_ = seq_[h,:,:]
        for s in range(seq_len-1):
            V1[s+1,h,:] = traj_[:,s+1] - traj_[:,s]
            fssa_weight1[s,h,h] = qz
    for s in range(seq_len-1):
        step_ = seq_[:,:,s]
        for h in range(max_nodes):
            for k in range(h,max_nodes):
                # l2_norm = anorm(step[h],step[k],V1[s, h], V1[s, k])
                # l2_norm2 = anorm(step[k],step[h],V1[s, k], V1[s, h])
                l2_norm = anorm1(step_[h],step_[k])
                l2_norm2 = anorm1(step_[k],step_[h])
                fssa_weight1[s,h,k] = l2_norm
                fssa_weight1[s,k,h] = l2_norm2
        if norm_lap_matr: 
            for i in range(len(step_)):
                for j in range(len(step_)):
                    if fssa_weight1[s,i,j] >0 :
                        if i!=j :
                            fssa_weight1[s,i,i] += qz-fssa_weight1[s,i,j]
                    elif i!=j:
                        fssa_weight1[s,i,j] = 0
            degree = np.array(fssa_weight1[s].sum(1))
            d_hat = np.diag(np.power(degree, -0.5).flatten())
            fssa_weight1[s] = d_hat.dot(fssa_weight1[s]).dot(d_hat)
            # G1 = nx.from_numpy_array(fssa_weight1[s,:,:])
            # fssa_weight1[s,:,:] = nx.normalized_laplacian_matrix(G1).toarray()
    return torch.from_numpy(fssa_weight1).type(torch.float)


def DIG(seq_, seq_rel, norm_lap_matr=True, qz=0.1):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]
    V = np.zeros((seq_len, max_nodes, 2)) 
    fssa_weight = np.zeros((seq_len, max_nodes, max_nodes))  

    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            fssa_weight[s, h, h] = qz  

            for k in range(h + 1, len(step_)):
                direction_hk = step_rel[k] - step_rel[h]  
                direction_kh = step_rel[h] - step_rel[k]  
                cosine_hk = get_cosine(step_[h], step_[k], direction_hk)
                cosine_kh = get_cosine(step_[k], step_[h], direction_kh)

                weight_hk = qz * (1 + cosine_hk) 
                weight_kh = qz * (1 + cosine_kh)

                fssa_weight[s, h, k] = weight_hk
                fssa_weight[s, k, h] = weight_kh

        if norm_lap_matr:
            for i in range(len(step_)):
                for j in range(len(step_)):
                    if fssa_weight[s, i, j] > 0:
                        if i != j:
                            fssa_weight[s, i, i] += qz - fssa_weight[s, i, j]
                    elif i != j:
                        fssa_weight[s, i, j] = 0
            degree = np.array(fssa_weight[s].sum(1))
            d_hat = np.diag(np.power(degree, -0.5).flatten())
            fssa_weight[s] = d_hat.dot(fssa_weight[s]).dot(d_hat)

    return torch.from_numpy(V).type(torch.float), torch.from_numpy(fssa_weight).type(torch.float)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True):
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        all_files = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if path[0] != "." and path.endswith(".txt")]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        fet_map = {}
        fet_list = []
        
        for path in all_files:
            # data = read_file(path, delim)
            # frames = np.unique(data[:, 0]).tolist()
            # hkl_path = os.path.splitext(path)[0] + ".pkl"  
            # with open(hkl_path, 'rb') as handle:
            #     new_fet = pickle.load(handle)
            # fet_map[hkl_path] = torch.from_numpy(new_fet)
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            hkl_path = os.path.splitext(path)[0] + ".pkl"  
            if os.path.exists(hkl_path):
                with open(hkl_path, 'rb') as handle:
                    new_fet = pickle.load(handle)
            else:
            # new_fet = np.zeros((14, 14, 512),dtype=np.float32)
                new_fet = np.zeros((data.shape[0], 2)) 
            # with open(hkl_path, 'rb') as handle:
            #     new_fet = pickle.load(handle)
            fet_map[hkl_path] = torch.from_numpy(new_fet)
            # new_fet = data[:, 2:]
            # fet_map[path] = torch.from_numpy(new_fet)
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))  

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))      
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, 
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))  
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),   
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])   
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq  #
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    fet_list.append(hkl_path)    

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        self.fet_map = fet_map
        self.fet_list = fet_list

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = [] 
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]
            v_,a_0 = PIG(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            a_1 = VIG(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            a_2= DIG(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            a_ = a_0 + a_1 + a_2
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_0 = PIG(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            a_1 = VIG(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            a_1 = DIG(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            a_ = a_0 + a_1+ a_2
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
            
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index],
            self.fet_map[self.fet_list[index]]
        ]
        return out
