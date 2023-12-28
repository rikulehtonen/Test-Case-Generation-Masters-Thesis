import sys
import os
sys.path.insert(0, '../../ATAG/')

from atag import Atag
from browserenv import BrowserEnv
from config.atag_config_2_modified import Atag_config
from browserenv.datahandler import TrainingData

trainingData = TrainingData({'training_data_path': 'config/temp/', 'filename': f'ppo_tc2_training_data_modified_1.json'})

#Parameters for Atag & algorithm
parameters = {
    'log_to_wandb': True,
    'lr': 5.558e-4,
    'entropy_coeff': 0.00,
    'max_timesteps': 100,
    'batch_timesteps': 20,
    'episode_max_timesteps': 20,
    'iteration_epochs': 4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip': 0.183,
    'save_frequency': 50,
    'name': 'PPO_TC2_MODIFIED',
    'trainingData': trainingData,
    'actor_file': 'final-models/ppo_pretrained_actor.pt',
    'critic_file': 'final-models/ppo_pretrained_critic.pt'
}

def main():
    # Config for browserEnvironment
    config = Atag_config()
    # Test environment for Browser library
    browserEnv = BrowserEnv(config)
    # ML algorithm for test generation
    atag_browser = Atag(env=browserEnv, **parameters)

    # Train 
    atag_browser.train()
    browserEnv.terminate()

if __name__ == '__main__':
    main()