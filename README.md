# banana-collector-rl

Current projects trains a DQN to solve Banana collector unity environment.

## Project Details
### The Environment
States are defined in a 37-dimensional space. Action space size is 4:
- FORWARD (0)
- BACK (1)
- LEFT (2)
- RIGHT (3)

### The task
The task is episodic, meaning that it has a defined end state (marked by done flag coming from environment). The task is considered to be solved when the agent achieves an average reward of +13 over 100 consecutive episodes.

## Getting started
. Fork the `banana-collector-rl` repo on GitHub.

1. Clone your fork locally:

```bash
cd <directory_in_which_repo_should_be_created>
git clone git@github.com:YOUR_NAME/banana-collector-rl.git
```

2. Now we need to install the environment. Navigate into the directory

```bash
cd banana-collector-rl
```

3. Then, install the environment with:

```bash
uv sync
```

4. Place Banana.app from course resources inside **banana_collector_rl**

## Instructions
Once environment has been activated, training can be run from command line:

```bash
uv run train.py
```

Running the module as it is will run the training loop with default parameters. These default parameters produce an agent that solves the environment. However, agent hyperparameters can be configured by means of runtime arguments:

#### `-i`, `--iterations`
- **Description**: Number of environment steps to train for.
- **Default**: Value from `config["DEFAULT"]["ITERATIONS"]`.

#### `-b`, `--batch_size`
- **Description**: Batch size for learning steps.
- **Default**: Value from `config["DEFAULT"]["BATCH_SIZE"]`.

#### `-g`, `--gamma`
- **Description**: Discount rate used in reinforcement learning.
- **Default**: Value from `config["DEFAULT"]["GAMMA"]`.

#### `-l`, `--learning_rate`
- **Description**: Learning rate for the optimizer.
- **Default**: Value from `config["DEFAULT"]["LEARNING_RATE"]`.

#### `-d`, `--eps_decay`
- **Description**: Decay of epsilon after each iteration.
- **Default**: Value from `config["DEFAULT"]["EPSILON_DECAY"]`.

#### `-e`, `--eps_end`
- **Description**: Minimum value of epsilon allowed.
- **Default**: Value from `config["DEFAULT"]["EPSILON_END"]`.
