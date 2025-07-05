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
from SFM import *

import torch.optim as optim


class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        self.pattn=SAM()
        self.embedding=nn.Linear(10,5)
    def forward(self, x, A,sequential_scene_attention):
        assert A.size(0) == self.kernel_size
        T=x.size(2)
        x=x.permute(0,2,3,1)  
        x=torch.cat((x,sequential_scene_attention),3)
        unified_graph=self.embedding(x.view(-1,10))
        unified_graph=unified_graph.view(1,T,A.size(2),-1)
        unified_graph=unified_graph.permute(0,3,1,2)
        unified_graph = self.conv(unified_graph)
        gcn_output_features = torch.einsum('nctv,tvw->nctw', (unified_graph, A))
        return gcn_output_features.contiguous(), A
    

class SAM_conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(SAM_conv,self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        gcn_in_channels=5
        self.gcn = ConvTemporalGraphical(gcn_in_channels, out_channels,
                                         kernel_size[1])
        self.scene_att=SAM()
        self.embedding=nn.Linear(10,5)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A,vgg):
        coordinates=x[:,:,-1,:]
        T=x.size(2)
        coordinates=coordinates.permute(0,2,1)  
        sequential_scene_attention=self.scene_att(vgg,coordinates)     
        sequential_scene_attention=sequential_scene_attention.unsqueeze(0)   
        sequential_scene_attention=sequential_scene_attention.unsqueeze(1)   
        sequential_scene_attention=sequential_scene_attention.repeat(1,T,1,1)

        res = self.residual(x)
        gcn_output_features, A = self.gcn(x, A,sequential_scene_attention)

        gcn_output_features = self.tcn(gcn_output_features) + res
        
        if not self.use_mdn:
            gcn_output_features = self.prelu(gcn_output_features)

        return gcn_output_features, A
def make_mlp(dim_list):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)
class SAM(nn.Module):
    def __init__(self,attn_L=196,attn_D=512,ATTN_D_DOWN=16,bottleneck_dim=8,embedding_dim=10):
        super(SAM, self).__init__()

        self.L = attn_L  
        self.D = attn_D  
        self.D_down = ATTN_D_DOWN  
        self.bottleneck_dim = bottleneck_dim  
        self.embedding_dim = embedding_dim   

        self.spatial_embedding = nn.Linear(2, self.embedding_dim)    
        self.pre_att_proj = nn.Linear(self.D, self.D_down)       

        mlp_pre_dim = self.embedding_dim + self.D_down  
        mlp_pre_attn_dims = [mlp_pre_dim, 512, self.bottleneck_dim]
        self.mlp_pre_attn = make_mlp(mlp_pre_attn_dims)  

        self.attn = nn.Linear(self.L*self.bottleneck_dim, self.L)    

    def forward(self, vgg, end_pos):

        npeds = end_pos.size(1)    
        end_pos = end_pos[0, :, :]     
        curr_rel_embedding = self.spatial_embedding(end_pos)  
        curr_rel_embedding = curr_rel_embedding.view(-1, 1, self.embedding_dim).repeat(1, self.L, 1)  
        vgg=vgg.repeat(npeds,1,1,1)     
        vgg = vgg.view(-1, self.D)   
        features_proj = self.pre_att_proj(vgg)       
        features_proj = features_proj.view(-1, self.L, self.D_down)  

        mlp_h_input = torch.cat([features_proj, curr_rel_embedding], dim=2) 
        attn_h = self.mlp_pre_attn(mlp_h_input.view(-1, self.embedding_dim+self.D_down))  
        attn_h = attn_h.view(npeds, self.L, self.bottleneck_dim)  

        attn_w = Func.softmax(self.attn(attn_h.view(npeds, -1)), dim=1) 
        attn_w = attn_w.view(npeds, self.L, 1)     

        sequential_scene_attention = torch.sum(attn_h * attn_w, dim=1)     
        return sequential_scene_attention 
class SIT(nn.Module):
    def __init__(self,stgcn_num =1,tcn_num=5,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(SIT,self).__init__()
        self.stgcn_num= stgcn_num
        self.tcn_num = tcn_num

        self.SAM_conv = nn.ModuleList()
        self.SAM_conv.append(SAM_conv(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.stgcn_num):
            self.SAM_conv.append(SAM_conv(output_feat,output_feat,(kernel_size,seq_len)))
        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
        for j in range(1,self.tcn_num):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)
            
            
        self.prelus = nn.ModuleList()
        for j in range(self.tcn_num):
            self.prelus.append(nn.PReLU())


        
    def forward(self,v,a,vgg):

        for k in range(self.stgcn_num):
            gcn_output_features,a = self.SAM_conv[k](v,a,vgg)
            
        gcn_output_features = gcn_output_features.view(gcn_output_features.shape[0],gcn_output_features.shape[2],gcn_output_features.shape[1],gcn_output_features.shape[3])
        
        gcn_output_features = self.prelus[0](self.tpcnns[0](gcn_output_features))

        for k in range(1,self.tcn_num-1):
            tcn_output_features =  self.prelus[k](self.tpcnns[k](gcn_output_features)) + gcn_output_features
            
        tcn_output_features = self.tpcnn_ouput(tcn_output_features)
        tcn_output_features = tcn_output_features.view(tcn_output_features.shape[0],tcn_output_features.shape[2],tcn_output_features.shape[1],tcn_output_features.shape[3])
        
        
        return tcn_output_features,a


