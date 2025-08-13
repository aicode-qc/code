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

import torch.optim as optim


class SocialForceModel(nn.Module):
    """
    社会力模型(Social Force Model, SFM)实现
    结合三个子图关系(位置、速度、方向)来模拟行人之间的交互力
    """
    def __init__(self, tau=0.5, A=2000, B=0.08, k=1.2e5, kappa=2.4e5):
        super(SocialForceModel, self).__init__()
        # SFM参数
        self.tau = tau  # 松弛时间
        self.A = A      # 排斥力强度
        self.B = B      # 排斥力范围
        self.k = k      # 接触力刚度
        self.kappa = kappa  # 摩擦力系数
        
        # 可学习的权重参数
        self.w_goal = nn.Parameter(torch.tensor(1.0))     # 目标力权重
        self.w_social = nn.Parameter(torch.tensor(1.0))   # 社会力权重
        self.w_position = nn.Parameter(torch.tensor(1.0)) # 位置子图权重
        self.w_velocity = nn.Parameter(torch.tensor(0.8)) # 速度子图权重
        self.w_direction = nn.Parameter(torch.tensor(0.6)) # 方向子图权重
        
    def compute_goal_force(self, current_pos, current_vel, desired_vel):
        """
        计算目标力：引导行人朝向目标方向
        """
        goal_force = (desired_vel - current_vel) / self.tau
        return goal_force
    
    def compute_social_force(self, positions, velocities, A_pos, A_vel, A_dir):
        """
        计算社会力：基于三个子图的行人间交互力
        positions: [batch, num_peds, 2] 位置
        velocities: [batch, num_peds, 2] 速度
        A_pos, A_vel, A_dir: 位置、速度、方向子图邻接矩阵
        """
        batch_size, num_peds, _ = positions.shape
        social_forces = torch.zeros_like(velocities)
        
        for i in range(num_peds):
            for j in range(num_peds):
                if i != j:
                    # 位置差向量
                    r_ij = positions[:, i, :] - positions[:, j, :]
                    d_ij = torch.norm(r_ij, dim=1, keepdim=True) + 1e-8
                    n_ij = r_ij / d_ij  # 单位方向向量
                    
                    # 速度差
                    v_ij = velocities[:, i, :] - velocities[:, j, :]
                    
                    # 基于三个子图的权重融合
                    weight_pos = A_pos[:, i, j].unsqueeze(-1) * self.w_position
                    weight_vel = A_vel[:, i, j].unsqueeze(-1) * self.w_velocity  
                    weight_dir = A_dir[:, i, j].unsqueeze(-1) * self.w_direction
                    
                    combined_weight = weight_pos + weight_vel + weight_dir
                    
                    # 排斥力计算
                    repulsive_force = self.A * torch.exp(-d_ij / self.B) * n_ij
                    
                    # 接触力(当距离很小时)
                    contact_force = torch.where(
                        d_ij < 0.5,  # 接触阈值
                        self.k * (0.5 - d_ij) * n_ij,
                        torch.zeros_like(n_ij)
                    )
                    
                    # 摩擦力
                    t_ij = torch.stack([-n_ij[:, 1], n_ij[:, 0]], dim=1)  # 切向量
                    friction_force = torch.where(
                        d_ij < 0.5,
                        self.kappa * (0.5 - d_ij) * torch.sum(v_ij * t_ij, dim=1, keepdim=True) * t_ij,
                        torch.zeros_like(t_ij)
                    )
                    
                    # 总的社会力
                    total_force = (repulsive_force + contact_force + friction_force) * combined_weight
                    social_forces[:, i, :] += total_force
                    
        return social_forces
    
    def forward(self, positions, velocities, desired_velocities, A_pos, A_vel, A_dir):
        """
        前向传播：计算总的社会力
        """
        # 目标力
        goal_forces = self.compute_goal_force(positions, velocities, desired_velocities)
        
        # 社会力
        social_forces = self.compute_social_force(positions, velocities, A_pos, A_vel, A_dir)
        
        # 总力
        total_forces = self.w_goal * goal_forces + self.w_social * social_forces
        
        return total_forces, goal_forces, social_forces




class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True,
                 use_sfm=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.use_sfm = use_sfm
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        self.pattn=SequentialSceneAttention()
        
        # 根据是否使用SFM调整embedding维度
        if self.use_sfm:
            self.embedding=nn.Linear(12,5)  # 10 + 2 (SFM force components)
            self.sfm = SocialForceModel()
            self.force_embedding = nn.Linear(2, 2)  # 力向量嵌入
        else:
            self.embedding=nn.Linear(10,5)
            
    def forward(self, x, A, sequential_scene_attention, A_subgraphs=None):
        assert A.size(0) == self.kernel_size
        T=x.size(2)
        x=x.permute(0,2,3,1)  
        
        # 如果使用SFM且提供了子图
        if self.use_sfm and A_subgraphs is not None:
            A_pos, A_vel, A_dir = A_subgraphs
            
            # 从轨迹数据中提取位置和速度
            positions = x[:, :, :, :2]  # 前两个维度是位置
            velocities = x[:, :, :, 2:4] if x.size(-1) >= 4 else torch.zeros_like(positions)  # 速度
            
            # 计算期望速度(简化为朝向最后一个位置的方向)
            if T > 1:
                desired_velocities = positions[:, -1:, :, :] - positions[:, -2:-1, :, :]
            else:
                desired_velocities = torch.zeros_like(velocities[:, -1:, :, :])
            desired_velocities = desired_velocities.expand_as(velocities)
            
            # 计算社会力
            sfm_forces = []
            for t in range(T):
                pos_t = positions[:, t, :, :]
                vel_t = velocities[:, t, :, :]
                des_vel_t = desired_velocities[:, t, :, :]
                
                # 确保子图矩阵维度正确
                if len(A_pos.shape) == 4:  # [batch, time, num_peds, num_peds]
                    A_pos_t = A_pos[:, t, :, :] if A_pos.size(1) > t else A_pos[:, -1, :, :]
                    A_vel_t = A_vel[:, t, :, :] if A_vel.size(1) > t else A_vel[:, -1, :, :]
                    A_dir_t = A_dir[:, t, :, :] if A_dir.size(1) > t else A_dir[:, -1, :, :]
                else:  # [batch, num_peds, num_peds]
                    A_pos_t = A_pos
                    A_vel_t = A_vel
                    A_dir_t = A_dir
                
                forces, _, _ = self.sfm(pos_t, vel_t, des_vel_t, A_pos_t, A_vel_t, A_dir_t)
                sfm_forces.append(forces)
            
            sfm_forces = torch.stack(sfm_forces, dim=1)  # [batch, time, num_peds, 2]
            
            # 嵌入力向量
            embedded_forces = self.force_embedding(sfm_forces)
            
            # 将力信息与原始特征连接
            x = torch.cat((x, embedded_forces), dim=3)
            embedding_input = torch.cat((x, sequential_scene_attention), 3)
            unified_graph = self.embedding(embedding_input.view(-1, 12))
        else:
            # 原始处理方式
            embedding_input = torch.cat((x, sequential_scene_attention), 3)
            unified_graph = self.embedding(embedding_input.view(-1, 10))
        
        unified_graph = unified_graph.view(1, T, A.size(2), -1)
        unified_graph = unified_graph.permute(0, 3, 1, 2)
        unified_graph = self.conv(unified_graph)
        gcn_output_features = torch.einsum('nctv,tvw->nctw', (unified_graph, A))
        return gcn_output_features.contiguous(), A
    

class SceneAttentionShare(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True,
                 use_sfm=True):
        super(SceneAttentionShare,self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        self.use_sfm = use_sfm
        gcn_in_channels=5
        self.gcn = ConvTemporalGraphical(gcn_in_channels, out_channels,
                                         kernel_size[1], use_sfm=use_sfm)
        self.scene_att=SequentialSceneAttention()
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

    def forward(self, x, A, vgg, A_subgraphs=None):
        coordinates=x[:,:,-1,:]
        T=x.size(2)
        coordinates=coordinates.permute(0,2,1)  
        sequential_scene_attention=self.scene_att(vgg,coordinates)     
        sequential_scene_attention=sequential_scene_attention.unsqueeze(0)   
        sequential_scene_attention=sequential_scene_attention.unsqueeze(1)   
        sequential_scene_attention=sequential_scene_attention.repeat(1,T,1,1)

        res = self.residual(x)
        
        # 传递子图信息给GCN层
        if self.use_sfm and A_subgraphs is not None:
            gcn_output_features, A = self.gcn(x, A, sequential_scene_attention, A_subgraphs)
        else:
            gcn_output_features, A = self.gcn(x, A, sequential_scene_attention)

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
class SequentialSceneAttention(nn.Module):
    def __init__(self,attn_L=196,attn_D=512,ATTN_D_DOWN=16,bottleneck_dim=8,embedding_dim=10):
        super(SequentialSceneAttention, self).__init__()

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
class SocialSoftAttentionGCN(nn.Module):
    def __init__(self,stgcn_num =1,tcn_num=5,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3,use_sfm=True):
        super(SocialSoftAttentionGCN,self).__init__()
        self.stgcn_num= stgcn_num
        self.tcn_num = tcn_num
        self.use_sfm = use_sfm
                
        self.SceneAttentionShares = nn.ModuleList()
        self.SceneAttentionShares.append(SceneAttentionShare(input_feat,output_feat,(kernel_size,seq_len),use_sfm=use_sfm))
        for j in range(1,self.stgcn_num):
            self.SceneAttentionShares.append(SceneAttentionShare(output_feat,output_feat,(kernel_size,seq_len),use_sfm=use_sfm))
        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
        for j in range(1,self.tcn_num):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)
            
            
        self.prelus = nn.ModuleList()
        for j in range(self.tcn_num):
            self.prelus.append(nn.PReLU())


        
    def forward(self, v, a, vgg, A_subgraphs=None):
        
        for k in range(self.stgcn_num):
            if self.use_sfm and A_subgraphs is not None:
                gcn_output_features, a = self.SceneAttentionShares[k](v, a, vgg, A_subgraphs)
            else:
                gcn_output_features, a = self.SceneAttentionShares[k](v, a, vgg)
            v = gcn_output_features  # 更新v用于下一层
            
        gcn_output_features = gcn_output_features.view(gcn_output_features.shape[0],gcn_output_features.shape[2],gcn_output_features.shape[1],gcn_output_features.shape[3])
        
        gcn_output_features = self.prelus[0](self.tpcnns[0](gcn_output_features))

        for k in range(1,self.tcn_num-1):
            tcn_output_features =  self.prelus[k](self.tpcnns[k](gcn_output_features)) + gcn_output_features
            gcn_output_features = tcn_output_features  # 更新用于下一层
            
        tcn_output_features = self.tpcnn_ouput(tcn_output_features)
        tcn_output_features = tcn_output_features.view(tcn_output_features.shape[0],tcn_output_features.shape[2],tcn_output_features.shape[1],tcn_output_features.shape[3])
        
        
        return tcn_output_features,a


