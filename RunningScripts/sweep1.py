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
    "name": "fast_test",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
    	"total_epochs" : {1},
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["LSTM"]},
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
