import random
from envs.sumo_files_marl.env.sim_env import TSCSimulator
from envs.sumo_files_marl.config import config

from gym import spaces
import numpy as np

import torch
import copy
import os 

# output_path


class SUMOEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    def __init__(self, args, rank):
        self.args = args
        id = args.seed + np.random.randint(0, 2023) + rank
        self.set_seed(id)
        # make env
        env_config = config['environment']
        # sumo_envs_num = len(env_config['sumocfg_files'])
        sumo_cfg = args.sumocfg_files
        sumo_cfg = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + sumo_cfg    
        env_config = copy.deepcopy(env_config)
        env_config['sumocfg_file'] = sumo_cfg
        port = args.port_start + id
        # print('------------------------', port )
        print('----port--', port, '----sumo_cfg--', sumo_cfg)
        output_path = str(self.args.run_dir) + '/'
        if args.use_eval:
            env_config['is_record'] = True
        self.env = TSCSimulator(env_config, port, output_path=output_path)
        self.ts_ids = self.env.all_tls
        self.n_actions = env_config['num_actions']
        self.obs_shape = env_config['obs_shape']
        self.iter_duration = env_config['iter_duration']
        self.yellow_duration = env_config['yellow_duration']
        self.episode_length_time = env_config['episode_length_time']
        self.sample_interval = env_config['sample_interval']
        self.unava_phase_index = []
        for i in self.env.all_tls:
            self.unava_phase_index.append(self.env._crosses[i].unava_index)
        self.num_agents = len(self.unava_phase_index)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for idx in range(self.num_agents):
            self.action_space.append(spaces.Discrete(n=env_config['num_actions']))
            self.share_observation_space.append(spaces.Box(-float('inf'), float('inf'), [env_config['obs_shape']*self.num_agents], dtype=np.float32))
            self.observation_space.append(spaces.Box(-float('inf'), float('inf'), [env_config['obs_shape']], dtype=np.float32))
        self.historical_obs = None

    def get_unava_phase_index(self):
        return np.array(self.unava_phase_index, dtype=object)
    
    def get_avail_agent_actions(self, agent_id):
        """
        input: agent_id, int
        output: the avail action index of agent_id, list, like: [1, 0, 1, ...]
        """
        unava_action = self.get_unava_phase_index()[agent_id]
        avail_actions = [1] * self.n_actions
        for i in unava_action:
            avail_actions[i] = 0
        return avail_actions

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def get_reward(self, reward, all_tls):
        ans = []
        for i in all_tls:
            ans.append(sum(reward[i].values()))
        return np.array(ans)
    
    def batch(self, env_output, use_keys, all_tl):
        all_agent_loc_list = []  
        """Transform agent-wise env_output to batch format."""
        if all_tl == ['gym_test']:
            return torch.tensor([env_output])
        obs_batch = {}
        for i in use_keys+['mask', 'neighbor_index', 'neighbor_dis']:
            obs_batch[i] = []
        is_loc = False
        if 'location' in use_keys:
            is_loc = True
            use_keys.remove('location')
        for agent_id in all_tl:
            out = env_output[agent_id]
            tmp_dict = {k: np.zeros(8) for k in use_keys}
            state, mask, neight_msg, location = out
            for i in range(len(state)):
                for s in use_keys:
                    tmp_dict[s][i] = state[i].get(s, 0) 
            for k in use_keys:
                obs_batch[k].append(tmp_dict[k])
            if is_loc:
                all_agent_loc_list.extend(list(location.values()))
                obs_batch['location'].append(list(location.values()) + [0] * 6)
            obs_batch['mask'].append(mask)
            obs_batch['neighbor_index'].append(neight_msg[0][0])
            obs_batch['neighbor_dis'].append(neight_msg[0][1])
        if is_loc:
            use_keys.append('location')
            all_agent_loc_list = np.array(all_agent_loc_list)
            min_val = all_agent_loc_list.min()
            max_val = all_agent_loc_list.max()
            normalized_loc = (all_agent_loc_list - min_val) / (max_val - min_val)  
            count = 0
            for i in range(len(all_tl)):
                obs_batch['location'][i][0] = normalized_loc[count]
                count += 1
                obs_batch['location'][i][1] = normalized_loc[count]
                count += 1
        for key, val in obs_batch.items():
            if key not in ['current_phase', 'mask', 'neighbor_index']:
                obs_batch[key] = torch.FloatTensor(np.array(val))
            else:
                obs_batch[key] = torch.LongTensor(np.array(val))
        if self.args.use_pressure:
            obs_batch['pressure'] = obs_batch.pop('pressure')
        else:
            obs_batch.pop('pressure')
        if self.args.use_gat:
            obs_batch['neighbor_index'] = obs_batch.pop('neighbor_index')
            obs_batch['neighbor_dis'] = obs_batch.pop('neighbor_dis')
        else:
            obs_batch.pop('neighbor_index')
            obs_batch.pop('neighbor_dis')
        obs_batch['mask'] = obs_batch.pop('mask') 
        self.obs_keys = list(obs_batch.keys())   
        obs_values = np.hstack(list(obs_batch.values())) 
        return obs_values

    def get_state(self):
        obs = self.env._get_state()
        obs_values = self.batch(obs, config['environment']['state_key'], self.env.all_tls)
        obs_values = self._obs_wrapper(obs_values)
        return obs_values, self.historical_obs
    
    def reset(self):
        info = {}
        his_obs = []
        for i in range(1, self.iter_duration + self.yellow_duration - self.sample_interval, self.sample_interval):
            his_obs.append(np.zeros((self.num_agents, self.obs_shape)))
        obs = self.env.reset()
        obs_values = self.batch(obs, config['environment']['state_key'], self.env.all_tls)
        obs_values = self._obs_wrapper(obs_values)
        his_obs.append(obs_values)
        self.historical_obs = np.stack(his_obs, axis=0)
        info['historical_obs'] = self.historical_obs
        return obs_values, info

    def step(self, action):
        tl_action_select = {}
        for tl_index in range(len(self.env.all_tls)):
            tl_action_select[self.env.all_tls[tl_index]] = \
                (self.env._crosses[self.env.all_tls[tl_index]].green_phases)[action[tl_index]]
        obs, reward, done, info = self.env.step(tl_action_select)
        obs = self.batch(obs, config['environment']['state_key'], self.env.all_tls)
        obs = self._obs_wrapper(obs)
        reward = self.get_reward(reward, self.env.all_tls)
        reward = reward.reshape(self.num_agents, 1)
        done = np.array([done] * self.num_agents)
        temp = []
        for item in info['historical_obs']:
            his_obs = self.batch(item, config['environment']['state_key'], self.env.all_tls)
            temp.append(his_obs)
        self.historical_obs = np.stack(temp, axis=0)
        info['historical_obs'] = self.historical_obs
        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.close()

    def _obs_wrapper(self, obs):
        if self.num_agents == 1:
            return obs[np.newaxis, :]
        else:
            return obs
