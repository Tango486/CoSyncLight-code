#!/usr/bin/env python3
# encoding: utf-8
 
name = 'test_freeze'
config = {
    "episode": {
        "num_train_rollouts": 100000,
        "rollout_length": 240,  
        "warmup_ep_steps": 0,
        "test_num_eps": 50
    },
    "agent": {
        "agent_type": "ppo",
        "single_agent": True
    },
    "ppo": {
        "gae_tau": 0.85,
        "entropy_weight": 0.01,
        "minibatch_size": 128,
        "optimization_epochs": 4,
        "ppo_ratio_clip": 0.2,
        "discount": 0.99,
        "learning_rate": 1e-4,
        "clip_grads": True,
        "gradient_clip": 2.0,
        "value_loss_coef": 1.0,
        'target_kl': None
    },
    "environment": {
        "num_actions": 8,
        # "obs_shape": 56,   ### ippo
        # "obs_shape": 56, #### gat
        "obs_shape": 48,
        "sample_interval": 2,
        
        "action_type": "select_phase",
        # "gui": True,
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,  
        "episode_length_time": 3600,
        "is_record": False,
        'output_path': None,
        "name": name,
        
        
        
        # 'port_start': 16900, # grid4x4
        # 'port_start': 16600, # sumo_fenglin_base_road
        # 'port_start': 16400, # nanshan
        # 'port_start': 16200, # arterial4x4
        # 'port_start': 16100, # ingolstadt21
        # 'port_start': 16000, # cologne8
        
        # 'port_start': 14300, # grid4x4  -------
        # 'port_start': 14800, # sumo_fenglin_base_road
        # 'port_start': 14400, # nanshan
        # 'port_start': 14200, # arterial4x4
        # 'port_start': 14100, # ingolstadt21
        # 'port_start': 14900, # cologne8
        
        # 'port_start': 17300, # grid4x4  -------
        # 'port_start': 17800, # sumo_fenglin_base_road
        # 'port_start': 17400, # nanshan
        # 'port_start': 18200, # arterial4x4
        # 'port_start': 17100, # ingolstadt21
        'port_start': 19900, # cologne8
        "sumocfg_files": [
            # "sumo_files/scenarios/large_grid2/exp_0.sumocfg",
            # "sumo_files/scenarios/large_grid2/exp_1.sumocfg",
            # # # "sumo_files/scenarios/real_net/most_0.sumocfg",
            # "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",
            # # "sumo_files/scenarios/sumo_wj3/rl_wj.sumocfg",
            
            # 'sumo_files_marl/scenarios/resco_envs/grid4x4/grid4x4.sumocfg'
            # "sumo_files_marl/scenarios/sumo_fenglin_base_road/base.sumocfg"
            # "sumo_files_marl/scenarios/nanshan/osm.sumocfg"
            # 'sumo_files_marl/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg'
            # 'sumo_files_marl/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg'
            # 'sumo_files_marl/scenarios/resco_envs/cologne8/cologne8.sumocfg'
        ],

        "state_key": ['location', 'current_phase', 'queue_length', "occupancy", 'flow', 'pressure'],  
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure',]


    },
    "model_save": {
        "frequency": 200,
        "path": "envs/sumo_files_marl/tsc/{}".format(name)
    },
    "parallel": {
        "num_workers": 1
    }
}
