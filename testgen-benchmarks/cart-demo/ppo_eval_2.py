import sys
import os
import numpy as np
sys.path.insert(0, '../../ATAG/')

from atag import Atag
from browserenv import BrowserEnv
from config.atag_config_2 import Atag_config

#Parameters for Atag & algorithm
parameters = {
    'log_to_wandb': False,
    'lr': 0.0,
    'entropy_coeff': 0.00,
    'gamma': 0.00,
    'gae_lambda': 0.00,
    'batch_timesteps': 10,
    'episode_max_timesteps': 20,
    'actor_file': 'final-models/ppo_tc2_actor.pt',
    'critic_file': 'final-models/ppo_tc2_critic.pt'
}

def main():
    # Config for browserEnvironment
    config = Atag_config()
    # Test environment for Browser library
    browserEnv = BrowserEnv(config)
    # ML algorithm for test generation
    atag_browser = Atag(env=browserEnv, **parameters)

    # Train 
    results = atag_browser.evaluate()
    browserEnv.terminate()

    print("=" * 10)
    print("mean")
    print(np.mean(results))
    print("std")
    print(np.std(results))
    print("max")
    print(np.max(results))
    print(config.stepReachedCount)
    print("=" * 10)

if __name__ == '__main__':
    main()