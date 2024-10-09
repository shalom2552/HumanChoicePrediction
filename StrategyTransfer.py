import utils
import environments
from consts import *
from utils.functions import *
import wandb
from utils import personas
import argparse

parser = argparse.ArgumentParser(description='Hyperparameter tuning with wandb.')
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# General Features
parser.add_argument('--ENV_HPT_mode', type=str2bool, default=False, help='Enable/disable HPT mode')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--task', choices=["off_policy", "on_policy"], default="off_policy", help='Task')
parser.add_argument('--ENV_LEARNING_RATE', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--loss_weight_type', type=str, default="None", help='Loss weight type')
parser.add_argument('--save_artifacts', type=str2bool, default=True, help='Save artifacts flag')
parser.add_argument('--strategies', default=None, help='Which strategies are include the training data')
parser.add_argument('--total_epochs', type=int, default=15, help='Number of epochs during training')

# Input Features
parser.add_argument('--features', type=str, default="EFs", help='Input features')

# Architecture Features
parser.add_argument('--architecture', type=str, default="LSTM", help='Model architecture')
parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimensions')
parser.add_argument('--layers', type=int, default=2, help='Number of layers')
parser.add_argument('--transformer_nheads', type=int, default=4, help='Transformer heads')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')

# Human data
parser.add_argument('--human_train_size', type=int, default=210, help='Training set size (in No. of users)')
parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for sentimental analysis')
parser.add_argument('--threshold_adjustment', type=float, default=0.01, help='Threshold adjustment')

# Online Simulation
parser.add_argument('--online_sim_type', type=str, default="before_epoch", help='Online simulation type')
parser.add_argument('--basic_nature', type=int, default=13, help='Basic nature of simulation')
parser.add_argument('--online_simulation_factor', type=float, default=4, help='Online simulation factor')
parser.add_argument('--simulation_bot_per_user', type=int, default=6, help='Simulation bot per user')
parser.add_argument('--simulation_signal_error', type=float, default=0.3, help='Simulation signal error')
parser.add_argument('--simulation_user_improve', type=float, default=0.01,
                    help='Simulation user improvement rate - (eta from Shapira et al. 2023)')
parser.add_argument('--max_games', type=int, default=4, help='Maximum games between sim. user and bot where eta==0')
parser.add_argument('--zero_knowledge', type=str2bool, default=True, help='Zero knowledge flag')
parser.add_argument('--bots_per_user', type=int, default=6, help='Bots per user')
parser.add_argument('--SIMULATION_EFs_PATH', type=str, default="data/EFs_by_GPT35.csv", help='Simulation EFs path')
parser.add_argument('--favorite_topic_method', type=str, default="review", help='Favorite topic method')

# Advanced Projects' Features
parser.add_argument('--agent', type=str, default="modelbasedRL", help='Agent type')
parser.add_argument('--force_train', type=str2bool, default=True, help='Force training of environment')
parser.add_argument('--output_dim', type=int, default=2, help='Output dimension')
parser.add_argument('--offline_simulation_size', type=int, default=0,
                    help='Offline simulation set size (in No. of users)')
parser.add_argument('--OFFLINE_SIM_DATA_PATH', type=str, default="data/LLM_games_personas.csv", help='LLM data path')
parser.add_argument('--personas_balanced', type=str2bool, default=True, help='Personas balanced flag')
parser.add_argument('--personas_group_number', type=int, default=-1, help='Personas group number')


args = parser.parse_args()

main_run = wandb.init(project='NLP2024_PROJECT_206320772_313510679')
config = wandb.config

config.update(args.__dict__)

meta_features_map = {"features": {"EFs": {"FEATURES_PATH": config["SIMULATION_EFs_PATH"], "REVIEW_DIM": 37},
                                  "GPT4": {"FEATURES_PATH": "data/GPT4_PCA_36.csv", "REVIEW_DIM": 36},
                                  "BERT": {"FEATURES_PATH": "data/BERT_PCA_36.csv", "REVIEW_DIM": 36}},
                     "architecture": {"LSTM": {"use_user_vector": True},
                                      "transformer": {"use_user_vector": False}}
                     }

for meta_feature, meta_feature_map in meta_features_map.items():
    if config[meta_feature] not in meta_feature_map.keys():
        raise NotImplementedError(meta_feature)
    for config_feature, val in meta_feature_map[config[meta_feature]].items():
        config[config_feature] = val

if "LLM_USERS_PER_PERSONA" in config.keys():
    assert "offline_simulation_size" not in config.keys()
    groups = personas.get_personas_in_group(config.personas_group_number)
    config["offline_simulation_size"] = config["LLM_USERS_PER_PERSONA"] * len(groups)

if "online_simulation_factor" in config.keys():
    config["online_simulation_size"] = (config["offline_simulation_size"] + config["human_train_size"]) * config["online_simulation_factor"]

config["input_dim"] = config['REVIEW_DIM'] + STRATEGY_DIM
config["wandb_run_id"] = wandb.run.id

set_global_seed(config["seed"])

all_user_points = []
all_bot_points = []
hotels = utils.Hotels(config)

env_name = config["wandb_run_id"]

if config["architecture"] == "LSTM":
    env_model = environments.LSTM_env.LSTM_env(env_name, config=config)
elif config["architecture"] == "transformer":
    env_model = environments.transformer_env.transformer_env(env_name, config=config)
