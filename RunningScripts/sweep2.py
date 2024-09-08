import wandb
YOUR_WANDB_USERNAME = "shalom_and_amit"
project = "NLP2024_PROJECT_206320772_313510679"


command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]


sweep_config = {
    "name": "LSTM: SimFactor=0/4 for any features representation 2",
    "method": "grid",
    # "metric": {
    #     "goal": "maximize",
    #     "name": "AUC.test.max"
    # },
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["LSTM"]},
        "online_simulation_factor": {"values": [0, 1, 2, 3, 4]},
        "basic_nature": {"values": [17]},
        "seed": {"values": list(range(1, 4))},
        "features": {"values": ["EFs", "GPT4"]},
    },
    "command": command
}


# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
