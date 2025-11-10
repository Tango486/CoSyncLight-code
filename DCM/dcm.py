# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from DCM.stsgcn import STSGCN
from algorithms.utils.mlp import MLPBase2 # type: ignore
from algorithms.utils.util import init, check # type: ignore
import torch.nn.functional as F

class DynaCo(nn.Module):
   
    def __init__(self, stsgcn_config, num_agents, num_T, obs_shape, batch_size, num_of_out_features, device, args):
        super(DynaCo, self).__init__()

        self.module_type = stsgcn_config['module_type']  
        self.act_type = stsgcn_config['act_type'] 
        self.temporal_emb = stsgcn_config['temporal_emb'] 
        self.spatial_emb = stsgcn_config['spatial_emb']  
        self.batch_size = batch_size
        self.device = device
        self.num_of_vertices = num_agents
        self.num_of_features = obs_shape
        self.points_per_hour = num_T 
        self.num_for_predict = stsgcn_config['num_for_predict']  
        self.num_of_out_features = num_of_out_features
        self.filters = stsgcn_config['filters']  
        self.first_layer_embedding_size = stsgcn_config['first_layer_embedding_size']  
 
        self.first_embedding = MLPBase2(args, (self.num_of_features,),self.first_layer_embedding_size, device)
        self.ps_encoding = PositionalEncoding_Emb(
            d_model=self.first_layer_embedding_size,
            dropout=0,
            max_len=1000,
            device=device
        )
        self.stsgcn = STSGCN(self.batch_size, self.points_per_hour, self.num_of_vertices, self.first_layer_embedding_size,
                            self.filters, self.module_type, self.act_type, self.num_of_out_features, self.temporal_emb, 
                            self.spatial_emb, device=self.device, rho=1, predict_length=self.num_for_predict)

    def forward(self, x, cur_epi=None, his_hn=None, his_cn=None, use_eval=False):
        x = self.first_embedding(x)
        B, T, N, C = x.shape
        x = self.ps_encoding(x.reshape(B*T, N, C))
        x, att_score, adj, adj_logits, (h_n, c_n) = self.stsgcn(x.reshape(B, T, N, C), cur_epi, his_hn, his_cn, use_eval)
        adj = torch.cat(adj, dim=0)  
        adj_logits = torch.cat(adj_logits, dim=0)
        att_score = torch.cat(att_score, dim=0)
        return x, att_score, adj, adj_logits, (h_n, c_n)  
    
    def evaluate_actions(self, historical_obs, his_hn, his_cn):
        x, att_score, adj, new_adj_logits, (h_n, c_n) = self.forward(historical_obs, his_hn=his_hn, his_cn=his_cn)
        score_action_log_probs = F.log_softmax(new_adj_logits, dim=-1)
        score_dist_entropy = torch.distributions.Categorical(logits=new_adj_logits).entropy().mean()
        return x, att_score, score_action_log_probs, score_dist_entropy, new_adj_logits
  
def pretty_print_args(args):
    args_dict = vars(args)
    print("Configuration:")
    for key, value in (args_dict.items()):
        print(f"self.{key} = {value}")

class PositionalEncoding_Emb(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cpu'):
        super(PositionalEncoding_Emb, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        self.tpdv = dict(dtype=torch.float32, device=device)

    def forward(self, x): 
        self.pe = check(self.pe).to(**self.tpdv)    
        range_ = torch.arange(0, x.shape[1])  
        range_ = check(range_).to(**self.tpdv).long()
        x = x + self.pe(range_)
        return self.dropout(x)
 