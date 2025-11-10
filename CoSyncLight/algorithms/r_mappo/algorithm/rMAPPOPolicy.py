import torch
from algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic, R_Actor_Trans, R_Critic_Trans, R_Critic_Trans_all, R_Critic_all
from utils.util import update_linear_schedule
from algorithms.utils.util import init, check
from torch import nn

import numpy as np

class R_MAPPOPolicy_Trans:
    
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.tpdv = dict(dtype=torch.float32, device=device)
 
        self.actor = R_Actor_Trans(args, self.obs_space, self.act_space, self.device)
 
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.actor.to(device)
        self.critic.to(device)
 
        total_params = sum(p.numel() for p in self.actor.parameters())
        print(f"Total number of actor parameters: {total_params}")
        total_params = sum(p.numel() for p in self.critic.parameters())
        print(f"Total number of critic parameters: {total_params}")
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                    lr=self.critic_lr,
                                                    eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
        
       
    def lr_decay(self, episode, episodes):
      
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cur_epi, cent_obs, obs, historical_obs, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, trans_masks=None):
 
        if np.random.uniform() > self.args.epsilon:
            deterministic = True
 
        actions, action_log_probs, actor_features_new, agg_states, adj, adj_log_probs, (h_n, c_n), att_score = self.actor(
            obs, 
            historical_obs, 
            cur_epi,
            available_actions,
            deterministic, 
            )

        values, rnn_states_critic = self.critic(actor_features_new, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_critic, adj, adj_log_probs, actor_features_new, agg_states, (h_n, c_n), att_score

    def get_values(self, cent_obs, rnn_states_critic, masks):
 
        if self.args.use_K == 0:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        else:
            roll_outs = self.args.n_rollout_threads
            obs_ = np.array(np.split(cent_obs, roll_outs))
            obs_ = check(obs_).to(**self.tpdv)
            atten_scores = self.actor.model.compute_score(obs_)  
            batch_size_obs = obs_.shape[0]
            topk_index = torch.topk(atten_scores, k=self.args.use_K, dim=-1)[1]  
            batch_indices = torch.arange(self.args.num_agents)[None,:,None].repeat(batch_size_obs, 1, 1).to(self.device)  
            topk_index = torch.cat((topk_index, batch_indices), dim=-1) 
            
            values, _ = self.critic(cent_obs, rnn_states_critic, masks, topk_index.cpu())
            
        return values

    def get_values_three(self, cent_obs, obs, historical_obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, trans_masks=None):
        actions, action_log_probs, actor_features, agg_states, adj, adj_log_probs, (h_n, c_n), att_score = self.actor(
            obs,
            historical_obs,
            available_actions=available_actions,
            deterministic=deterministic,)
        values, _ = self.critic(actor_features, rnn_states_critic, masks)
        return values
    
    def evaluate_actions(self, cent_obs, obs, agg_states, his_hn, his_cn, historical_obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, score_batch=None, actor_features_batch=None, trans_masks=None):
    
        action_log_probs, dist_entropy, score_action_log_probs, score_dist_entropy, score_action_logits, att_score = self.actor.evaluate_actions(obs,
                                                                     agg_states,
                                                                     his_hn,
                                                                     his_cn,                                                                     
                                                                     historical_obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks,
                                                                     score_batch,
                                                                     trans_masks=trans_masks
                                                                     )
       
       
        values, _ = self.critic(actor_features_batch, rnn_states_critic, masks, backward=True)
        
        return values, action_log_probs, dist_entropy, score_action_log_probs, score_dist_entropy, score_action_logits, att_score
     
    def act(self, obs, historical_obs, available_actions=None, deterministic=True, trans_masks=None, use_eval = False):
    
        actions, action_log_probs, actor_features_new, agg_states, score, score_log_probs, (h_n, c_n), att_score = self.actor(obs, historical_obs,
                                                                 available_actions=available_actions,
                                                                 deterministic=deterministic, 
                                                                 use_eval = use_eval,
                                                                 )
        return actions, None, score, att_score
