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
    "name": "sentAnlysis final",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "seed": {"values": list(range(1, 4))},
        "threshold": {"values": [0.6]},
        "threshold_adjustment": {"values": [0.01]}    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
