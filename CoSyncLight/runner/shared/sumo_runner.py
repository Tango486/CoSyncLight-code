import time
import numpy as np
import torch
import wandb
import imageio
import os

from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    return x.detach().cpu().numpy()


class SUMORunner():
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       
        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.best_average_episode_rewards = self.search_best_reward()

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')  
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)

            self.save_dir = str(self.run_dir / 'models')  
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        
        from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy_Trans as Policy
        from algorithms.r_mappo.r_mappo import R_MAPPO_Trans as TrainAlgo

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],  
                            share_observation_space,  
                            self.envs.action_space[0],
                            device = self.device)


        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        self.load_model = self.all_args.load_model
        self.model_index = self.all_args.model_index
        if self.load_model:
            self.restore()
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        self.epsilon = self.all_args.epsilon
        self.anneal_epsilon = (self.all_args.epsilon - self.all_args.min_epsilon) / self.all_args.anneal_steps
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        print("total episode: ", episodes)
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            for step in range(self.episode_length):
                time1=time.time()

                print('-----step', step, end=" ", flush=True)
 
                values, actions, action_log_probs, rnn_states_critic, actions_env, adj, adj_log_probs, actor_features, agg_states, (h_n, c_n) = self.collect(step, episode) 
                # Obser reward and next obs
                ###### tsc specified actions
 
                obs, rewards, dones, infos = self.envs.step(actions.squeeze().tolist())
 
                obs = np.expand_dims(obs, axis=0)
                rewards = np.expand_dims(rewards, axis=0)
                dones = np.expand_dims(dones, axis=0)
 
                historical_obs = np.expand_dims(infos['historical_obs'], axis=0)
                available_actions = []
                for agent_id in range(self.num_agents):
                    available_actions.append(self.envs.get_avail_agent_actions(agent_id))
                available_actions = np.expand_dims(np.array(available_actions), axis=0) 
 
                data = obs, historical_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states_critic, available_actions, adj, adj_log_probs, actor_features, agg_states, h_n, c_n

                # insert data into buffer
                self.insert(data)

                ##### episide decay
                self.all_args.epsilon = self.all_args.epsilon - self.anneal_epsilon if self.all_args.epsilon > self.all_args.min_epsilon else self.all_args.epsilon
                time2=time.time()
                iteration_time = time2 - time1   
                print(f"Iteration  {iteration_time:.4f} seconds")
 
            self.warmup()
 
            self.compute() # Calculate returns for the collected data.

            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                float(total_num_steps / (end - start))))
                print("剩余时间: {}".format(((self.num_env_steps-total_num_steps)/float(total_num_steps / (end - start)))/3600))
                print("{} 剩余时间: {}".format(1000,((240000-total_num_steps)/float(total_num_steps / (end - start)))/3600))

                if self.env_name == "SUMO":
                    env_infos = {}
 
                    for info in [infos['all_reward']]:
                        for agent_id in range(self.num_agents):
                            for k, v in info[list(info.keys())[agent_id]].items():
                                if k not in env_infos:
                                    env_infos[k] = []
                                env_infos[k].append(v)  

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}, saved at {}".format(train_infos["average_episode_rewards"], self.log_dir + 'train_average_episode_rewards.txt'))
                self.write_to_file(train_infos["average_episode_rewards"], 'train_average_episode_rewards.txt')
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)
            if self.best_average_episode_rewards < train_infos["average_episode_rewards"]:
                self.best_average_episode_rewards = train_infos["average_episode_rewards"]
                self.save(episode, suffix='best')
 

    def warmup(self):
 
        obs, info = self.envs.reset()   
 
        historical_obs = info['historical_obs']
 
        obs = np.expand_dims(obs, 0)  
 
        if self.use_centralized_V: 
            share_obs = obs.reshape(self.n_rollout_threads, -1)  
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)  
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()   
        self.buffer.obs[0] = obs.copy() 
        self.buffer.historical_obs[0] = np.expand_dims(historical_obs, axis=0).copy()  
        available_actions = []
        for agent_id in range(self.num_agents):
            available_actions.append(self.envs.get_avail_agent_actions(agent_id))
        available_actions = np.expand_dims(np.array(available_actions), axis=0) 
        self.buffer.available_actions[0] = available_actions.copy()
        
    def get_ava_actions(self, ava):
        available_actions = np.ones((self.all_args.n_rollout_threads, self.all_args.num_agents, self.all_args.num_actions))
        if ava.all() == None:
            return available_actions
        if len(ava.shape) == 2: 
            for i in range(self.all_args.num_agents):
                for j in range(self.all_args.n_rollout_threads):
                    available_actions[j, i, ava[j][i]] = 0
        elif ava is not None and ava.shape[-1] != 0:
            for i in range(self.all_args.n_rollout_threads):
                for j in range(self.all_args.num_agents):
                    available_actions[i, j, ava[i][j][0]] = 0
            
        return available_actions
    
    
    @torch.no_grad()
    def collect(self, step, episode):  
        self.trainer.prep_rollout()
        
        if self.all_args.part_mask:
             trans_masks = np.concatenate(self.buffer.trans_masks[step-1])
        else:
             trans_masks = np.concatenate(self.buffer.trans_masks[step])
     
        cur_epi = episode*240 + step
        value, action, action_log_prob, rnn_states_critic, adj, adj_log_probs, actor_features, agg_states, (h_n, c_n), att_score = self.trainer.policy.get_actions(
            cur_epi,
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),  
            self.buffer.historical_obs[step], 
            np.concatenate(self.buffer.rnn_states_critic[step]), 
            np.concatenate(self.buffer.masks[step]),  
            available_actions=np.concatenate(self.buffer.available_actions[step]), 
            trans_masks = trans_masks  
            )

        if adj is not None:
            adj = np.array(np.split(_t2n(adj), self.n_rollout_threads))
            adj_log_probs = np.array(np.split(_t2n(adj_log_probs), self.n_rollout_threads))
        actor_features = np.array(np.split(_t2n(actor_features), self.n_rollout_threads))
        agg_states = np.array(_t2n(agg_states))
        h_n = np.array(_t2n(h_n))
        c_n = np.array(_t2n(c_n))
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)  
        else:
            raise NotImplementedError
        return values, actions, action_log_probs, rnn_states_critic, actions_env, adj, adj_log_probs, actor_features, agg_states, (h_n, c_n)
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
       
        next_values = self.trainer.policy.get_values_three(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.obs[-1]),
                                                self.buffer.historical_obs[-1],
                                                np.concatenate(self.buffer.rnn_states[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]),
                                                np.concatenate(self.buffer.available_actions[-1]),
                                                trans_masks=np.concatenate(self.buffer.trans_masks[-1])
                                                ) 
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def insert(self, data):
        obs ,historical_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states_critic, available_actions, adj, adj_log_probs, actor_features, agg_states, h_n, c_n = data

        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, historical_obs, rnn_states_critic, actions, action_log_probs, values, rewards, masks, available_actions=available_actions,
                           score=adj, score_log_probs=adj_log_probs, actor_features=actor_features, agg_states = agg_states, h_n = h_n, c_n = c_n)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

    def save(self, episode, suffix=None):
        if suffix != None:
            """Save policy's actor and critic networks."""
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_{}.pt".format(suffix))
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_{}.pt".format(suffix))
            if self.trainer._use_valuenorm: 
                policy_vnorm = self.trainer.value_normalizer
                torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm_{}.pt".format(suffix))
        else:
            """Save policy's actor and critic networks."""
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_{}.pt".format(episode%3))
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_{}.pt".format(episode%3))
            if self.trainer._use_valuenorm: 
                policy_vnorm = self.trainer.value_normalizer
                torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm_{}.pt".format(episode%3))
        
    def restore(self):
        """Restore policy's networks from a saved model."""
        if self.all_args.model_dir == None:
            model_dir = self.save_dir
        else:
            model_dir = self.all_args.model_dir
        if self.model_index == None:
            self.model_index = ''
        else:
            self.model_index = '_' + self.model_index
        print("!!!Restore policy's networks from {}!!!\n".format(model_dir))
        print("model index:", self.model_index)
        policy_actor_state_dict = torch.load(str(model_dir) + '/actor' + self.model_index + '.pt', map_location=self.device)
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:  
            policy_critic_state_dict = torch.load(str(model_dir) + '/critic' + self.model_index + '.pt', map_location=self.device)
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(str(model_dir) + '/vnorm' + self.model_index + '.pt', map_location=self.device)
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                if k == 'average_episode_rewards':
                    self.writter.add_scalars("", {k: v}, total_num_steps)
                else :
                    self.writter.add_scalars("log_data", {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars('', {k: np.mean(v)}, total_num_steps)
                    
    def write_to_file(self, reward: float, filename):
        """
        Write the reward value to a file in the results directory.

        Parameters:
        :reward: The reward value to be written to the file.
        :filename: The name of the file to which the reward will be written.

        This function appends the reward value to the specified file.
        """
        file_path = self.log_dir + '/' + filename
        with open(file_path, 'a') as f:
            f.write(f"{reward},")  
    def search_best_reward(self):
        reward_txt_dir = str(self.all_args.run_dir) + r'\logs\train_average_episode_rewards.txt'
        max_num = float('-inf') 
 
        if os.path.isfile(reward_txt_dir):
            with open(reward_txt_dir, 'r') as f:
                for line in f:
 
                    parts = line.strip().split(',')
                    for part in parts:
                        if part.strip() != '':
                            try:
                                num = float(part.strip())
                                if num > max_num:
                                    max_num = num
                            except ValueError:
                                continue  
        return max_num
