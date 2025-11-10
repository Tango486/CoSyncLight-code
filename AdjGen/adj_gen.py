import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

 
class AdjGen(nn.Module):
 
    def __init__(self, state_feature_size: int, dg_hidden_size: int, batch_size, num_of_vertices, device, hidden_state_lr = 0.5, gumbel_softmax = True, gumbel_tau = 1, sparse_threshold = 0):
 
        super(AdjGen, self).__init__()
        self.batch_size = batch_size
        self.N = num_of_vertices 
        self.device = device
        self.num_layers = 1
        self.num_directions = 2   
        self.lstm_hidden_dim = dg_hidden_size
        self.lstm = nn.LSTM(input_size = 2*state_feature_size, hidden_size = self.lstm_hidden_dim, num_layers = self.num_layers, batch_first=True, bidirectional=True)
        self.hard_encoding = nn.Sequential(
            nn.Linear(self.num_directions*self.lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.input_norm = nn.LayerNorm(2 * state_feature_size)
        self.output_norm = nn.LayerNorm(self.num_directions * self.lstm_hidden_dim)

        self.MAX_BATCH = 6*self.N*self.N  
        self.current_episode = 0
        self.recurrent_index = 0
        self.h_n = torch.zeros((self.num_layers*self.num_directions, self.MAX_BATCH, self.lstm_hidden_dim)).to(self.device)
        self.c_n = torch.zeros((self.num_layers*self.num_directions, self.MAX_BATCH, self.lstm_hidden_dim)).to(self.device)
 
        self.gumbel_softmax = gumbel_softmax
        self.gumbel_tau = gumbel_tau
        self.sparse_threshold = sparse_threshold

        self.tau = 1.0
        self.init_tau = 1.0
        self.tau_decay_rate = 1e-5
        self.min_tau = 0.1
    def forward(self, inputs, cur_epi, his_hn=None, his_cn=None, use_eval=False):
 
        B, num_win, N, C = inputs.shape  
        inputs = inputs.reshape(-1, *inputs.shape[2:])
        inputs_i = inputs.unsqueeze(2).expand(B*num_win, N, N, C) 
        inputs_j = inputs.unsqueeze(1).expand(B*num_win, N, N, C) 
        lstm_inputs = torch.cat([inputs_i, inputs_j], dim=-1)  
        lstm_inputs = lstm_inputs.reshape(B, num_win, *lstm_inputs.shape[1:]) 

        lstm_inputs = self.input_norm(lstm_inputs)
        lstm_inputs = lstm_inputs.view(lstm_inputs.shape[0], lstm_inputs.shape[1], -1, lstm_inputs.shape[-1]) 
        lstm_inputs = lstm_inputs.permute(0, 2, 1, 3) 
        lstm_inputs = lstm_inputs.reshape(-1, *lstm_inputs.shape[2:])  
 
        if his_hn == None:
            if self.recurrent_index < 3: 
                h_in = self.h_n[:, self.recurrent_index*1*self.N*self.N:(1+self.recurrent_index)*1*self.N*self.N, :].contiguous()
                c_in = self.c_n[:, self.recurrent_index*1*self.N*self.N:(1+self.recurrent_index)*1*self.N*self.N, :].contiguous()
                lstm_output, (h_n_out, c_n_out) = self.lstm(lstm_inputs, (h_in, c_in))
                
 
                self.h_n[:, self.recurrent_index*1*self.N*self.N:(1+self.recurrent_index)*1*self.N*self.N, :] = h_n_out
                self.c_n[:, self.recurrent_index*1*self.N*self.N:(1+self.recurrent_index)*1*self.N*self.N, :] = c_n_out

            else:
                h_in = self.h_n[:, self.recurrent_index*1*self.N*self.N:(1+self.recurrent_index)*1*self.N*self.N, :].contiguous()
                c_in = self.c_n[:, self.recurrent_index*1*self.N*self.N:(1+self.recurrent_index)*1*self.N*self.N, :].contiguous()
                lstm_output, (h_n_out, c_n_out) = self.lstm(lstm_inputs, (h_in, c_in))

                self.h_n[:, self.recurrent_index*1*self.N*self.N:(1+self.recurrent_index)*1*self.N*self.N, :] = h_n_out
                self.c_n[:, self.recurrent_index*1*self.N*self.N:(1+self.recurrent_index)*1*self.N*self.N, :] = c_n_out
  
        else:
            if self.recurrent_index < 3:
                his_hn = his_hn[:, :, self.recurrent_index * 1*self.N*self.N:(1 + self.recurrent_index) * 1*self.N*self.N, :]
                his_cn = his_cn[:, :, self.recurrent_index * 1*self.N*self.N:(1 + self.recurrent_index) * 1*self.N*self.N, :]
            else:
                his_hn = his_hn[:, :, self.recurrent_index * 1*self.N*self.N:(1 + self.recurrent_index) * 1*self.N*self.N, :]
                his_cn = his_cn[:, :, self.recurrent_index * 1*self.N*self.N:(1 + self.recurrent_index) * 1*self.N*self.N, :]

            T, B, H, D = his_hn.shape
            his_hn = his_hn.permute(1, 0, 2, 3).reshape(B, T * H, D).contiguous()
            his_cn = his_cn.permute(1, 0, 2, 3).reshape(B, T * H, D).contiguous()

            lstm_output, (his_hn_out, his_cn_out) = self.lstm(lstm_inputs, (his_hn, his_cn))
 
        lstm_output = self.output_norm(lstm_output.permute(1, 0, 2))
        lstm_output = lstm_output.view(inputs.shape[0], inputs.shape[1], inputs.shape[1], -1)  
        graph_adj_logits = self.hard_encoding(lstm_output)

        self.tau = max(self.init_tau * np.exp(-cur_epi * self.tau_decay_rate), self.min_tau) if cur_epi is not None else self.tau
        if use_eval:
            self.tau = self.min_tau
        graph_adj_sample = f.gumbel_softmax(graph_adj_logits, tau=self.tau, hard=True) 
        graph_adj = graph_adj_sample[:, :, :, 1] 

        graph_adj_logits = graph_adj_logits[:, :, :, 1]
 
        self.recurrent_index += 1
        if self.recurrent_index == 6:
            self.recurrent_index = 0
 
        return graph_adj, graph_adj_logits, (self.h_n, self.c_n) 
