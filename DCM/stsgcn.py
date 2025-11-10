import torch.nn as nn
import torch
import torch.nn.init as init

import numpy as np
import torch.nn.functional as f

from AdjGen.adjgen_config import adjgen_get_config
from AdjGen.adj_gen import AdjGen

class STSGCN(nn.Module):
 
    def __init__(self, batch_size, input_length, num_of_vertices, num_of_features, filter_list, module_type, activation, num_of_out_features,  temporal_emb=True, spatial_emb=True, device="cpu", rho=1, predict_length=12, ):
        super(STSGCN, self).__init__()
        self.batch_size = batch_size
        self.input_length = input_length
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filter_list = filter_list 
        self.module_type = module_type
        self.activation = activation
        self.temporl_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.device = device
        self.rho = rho
        self.predict_length = predict_length
        self.num_of_out_features = num_of_out_features
        data_length = self.input_length
        print("Data lenght is: {}, num of stsgcl layers: {}".format(data_length, len(self.filter_list)))
        for filters in self.filter_list:  
            data_length -= 2
            self.num_of_features = filters[-1]

        filters = self.filter_list[0]
        self.stsgcl = STSGCL(self.batch_size, data_length, self.num_of_vertices, self.num_of_features, filters, self.module_type, self.activation, self.temporl_emb, self.spatial_emb, self.device)
        self.output_layer = OUTPUT(self.num_of_vertices, data_length, self.num_of_features, num_of_filters=self.num_of_out_features, predict_length=1)

    def forward(self, x, cur_epi, his_hn, his_cn, use_eval=False):
        adj_list = []
        adj_logits_list = []
        att_score_list = []
        for _ in self.filter_list: 
            x, att_score, adj, adj_logits, (h_n, c_n) = self.stsgcl(x, cur_epi, his_hn, his_cn, use_eval)
            adj_list.append(adj) 
            adj_logits_list.append(adj_logits)
            att_score_list.append(att_score)
        need_concat = []
        for _ in range(self.predict_length):
            need_concat.append(
                self.output_layer(x)
            )
        x = torch.cat(need_concat, dim=1)
        return x, att_score_list, adj_list, adj_logits_list, (h_n, c_n)


class STSGCL(nn.Module):
  
    def __init__(self, batch_size, T, num_of_vertices, num_of_features, filters,  module_type, activation,
                        temporal_emb, spatial_emb, device):
        super(STSGCL, self).__init__()
        self.batch_size = batch_size
        self.T = T 
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.module_type = module_type
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.device = device
        self.stsgcm = STSGCM(self.batch_size, self.filters, self.num_of_features, self.num_of_vertices, self.activation, self.device)
        self.max_num_windows = 0
        assert self.module_type in {'sharing', 'individual'}
 
    def forward(self, x, cur_epi, his_hn, his_cn, use_eval=False): 
        if self.module_type == 'individual':
            return self.sthgcn_layer_individual(x, cur_epi, his_hn, his_cn, use_eval)
        else:
            pass
 
    def sthgcn_layer_individual(self, x, cur_epi, his_hn, his_cn, use_eval=False):
      
        B, self.T, N, C = x.shape
        x = self.position_embedding(x)
        window_size = 3
        stride = 1   
        valid_starts = [i for i in range(0, self.T - window_size + 1, stride)]
        num_windows = len(valid_starts)
        self.max_num_windows = num_windows if self.max_num_windows < num_windows else self.max_num_windows
     
        t_list = [x[:, i:i+window_size, :, :].unsqueeze(1) for i in range(0, self.T-2, stride)]
        t_all = torch.cat(t_list, dim=1)  
 
        t_all = t_all.reshape(t_all.shape[0], t_all.shape[1], window_size * N, C)  

        t_out, att_score, adj, adj_logits, (h_n, c_n) = self.stsgcm(t_all, cur_epi, his_hn, his_cn, use_eval)
 
        t_out = t_out.view(B, num_windows, N, -1)
 
        t_out = t_out.unsqueeze(2) 
 
        need_concat = [t_out[:, i] for i in range(num_windows)]  
 
        return torch.cat(need_concat, dim=1), att_score, adj, adj_logits, (h_n, c_n)
 

    def sthgcn_layer_sharing(self):
        pass
    def position_embedding(self, data):
        def xavier_init(tensor):
            init.xavier_uniform_(tensor, gain=0.0003)  
        temporal_emb_tensor = None
        spatial_emb_tensor = None
        if self.temporal_emb:
         
            temporal_emb_tensor = torch.empty(1, self.T, 1, self.num_of_features).to(self.device)
            xavier_init(temporal_emb_tensor)
        if self.spatial_emb:
         
            spatial_emb_tensor = torch.empty(1, 1, self.num_of_vertices, self.num_of_features).to(self.device)
            xavier_init(spatial_emb_tensor)
        if temporal_emb_tensor is not None:
            data = data + temporal_emb_tensor
        if spatial_emb_tensor is not None:
            data = data + spatial_emb_tensor
        return data

class STSGCM(nn.Module):  
    def __init__(self, batch_size, filters, num_of_features, num_of_vertices, activation, device):
        super(STSGCM, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.num_of_vertices = num_of_vertices

        self.filters = filters
        self.num_of_features = num_of_features
        self.activation = activation
        assert all(x == self.filters[0] for x in self.filters), "STSGCM中的filters元素需要相等!"
        output_dim = self.filters[0]
        self.gcn_operation = GCN_OPERATION(self.batch_size, output_dim, self.num_of_features,self.num_of_vertices, self.activation, self.device)
        self.num_of_features = output_dim
 
    def forward(self, x, cur_epi, his_hn, his_cn, use_eval=False):
        need_concat = []
 
        adj_list = []
        adj_logits_list = []
        att_score_list = []
        for _ in range(len(self.filters)): 
            x, att_score, adj, adj_logits, (h_n, c_n) = self.gcn_operation(x, cur_epi, his_hn, his_cn, use_eval)  
            adj_list.append(adj)
            adj_logits_list.append(adj_logits)
            att_score_list.append(att_score)
            x1 = x.reshape(-1, *x.shape[2:])
            need_concat.append(x1)  

 
        need_concat = [
            torch.unsqueeze(
                i[:, 2 * self.num_of_vertices:3 * self.num_of_vertices, :], dim=0   
            ) for i in need_concat
        ]
 
        return torch.max(torch.cat(need_concat, dim=0), dim=0)[0], torch.cat(att_score_list,dim=0), torch.cat(adj_list,dim=0), torch.cat(adj_logits_list,dim=0), (h_n, c_n)
    
class time_encode(nn.Module):
    def __init__(self, N, input_dim):
        super(time_encode, self).__init__()
        self.N = N
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim)
        self.fc3 = nn.Linear(self.input_dim, self.input_dim)
    def forward(self, x):
 
        t1 = x[:, 0: self.N, :]
        t2 = x[:, self.N: 2*self.N, :]
        t3 = x[:, 2*self.N: 3*self.N, :]
        x1 = self.fc1(t1)
        x2 = self.fc2(t2)
        x3 = self.fc3(t3)
        ans = torch.concat([x1, x2, x3], dim=1)
        return ans
    

class GCN_OPERATION(nn.Module):
    def __init__(self, batch_size, output_dim, num_of_features, num_of_vertices, activation, device):
        super(GCN_OPERATION, self).__init__()
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.device = device
        eg_config = adjgen_get_config
        
        self.adjgen = AdjGen(num_of_features, eg_config.hidden_state_features_size, 
                    self.batch_size, 3*self.num_of_vertices, device=self.device)
        
        self.activation = activation
        self.attention_dim = 128
 
        self.qkv_proj = nn.Linear(num_of_features, 3 * self.attention_dim)
        self.linear = nn.Linear(self.attention_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, cur_epi, his_hn, his_cn, use_eval=False):
 
        B, num_win, N, C = x.shape
 
        adj, adj_logits, (h_n, c_n) = self.adjgen(x, cur_epi, his_hn, his_cn, use_eval)   
 
        x = x.reshape(-1, *x.shape[2:])
        qkv = self.qkv_proj(x)  
        q, k, v = qkv.chunk(3, dim=-1)  
 
        scale = np.sqrt(self.attention_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale   
         
        attention_weights = f.softmax(scores, dim=-1) 
        attention_weights = attention_weights * adj  
 
        attention_sum = attention_weights.sum(dim=-1, keepdim=True)
        mask = (attention_sum > 0).float()
        normalized_weights = attention_weights / (attention_sum + 1e-8) * mask
 
        v = self.relu(v)
        output = torch.matmul(normalized_weights, v)  
 
        output = self.linear(output)  
 
        output = output.reshape(-1, num_win, *output.shape[1:])
        return output, normalized_weights, adj, adj_logits, (h_n, c_n)
 
class OUTPUT(nn.Module):
    def __init__(self, num_of_vertices, input_length, num_of_features, num_of_filters, predict_length=1):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.num_of_filters = num_of_filters
        self.predict_length = predict_length
        self.linear1 = nn.Linear(in_features=self.input_length * self.num_of_features, out_features=self.num_of_filters)
        self.activation = nn.ReLU()

    def forward(self, data):
        data = data.permute(0, 2, 1, 3)
        data = data.reshape(-1, self.num_of_vertices, self.input_length * self.num_of_features)
        data = self.activation(self.linear1(data))
        return data
    

