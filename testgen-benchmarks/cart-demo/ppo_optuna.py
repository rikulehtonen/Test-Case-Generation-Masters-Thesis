import sys
import os
sys.path.insert(0, '../../ATAG/')

import optuna
from atag import Atag
from browserenv import BrowserEnv
from config.atag_config import Atag_config
from browserenv.datahandler import TrainingData

def objective(trial):
    # Parameters to be tuned
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    entropy_coeff = trial.suggest_float('entropy_coeff', 0.0, 0.1)
    clip = trial.suggest_float('clip', 0.1, 0.3)

    # Fixed parameters
    fixed_parameters = {
        'log_to_wandb': False,
        'max_timesteps': 4,
        'batch_timesteps': 20,
        'episode_max_timesteps': 20,
        'iteration_epochs': 4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'save_frequency': 50,
        'actor_file': 'final-models/actor.pt',
        'critic_file': 'final-models/critic.pt'        
    }
    
    parameters = {
        'lr': lr,
        'entropy_coeff': entropy_coeff,
        'clip': clip,
        **fixed_parameters
    }

    trainingData = TrainingData({'training_data_path': 'config/temp/'})
    config = Atag_config()
    browserEnv = BrowserEnv(config)
    atag_browser = Atag(env=browserEnv, **parameters)
    
    metric= atag_browser.train()
    browserEnv.terminate()

    # Here you need to return the metric you want to optimize.
    # For example, it could be the loss or accuracy.
    # You need to modify the code to extract this metric from your model.
    return metric

def main():
    study = optuna.create_study(direction='maximize', storage="sqlite:///db.sqlite3", load_if_exists=True, study_name="ppo_cart")
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    main()
