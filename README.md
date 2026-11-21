![Cliff Walking](https://user-images.githubusercontent.com/53657825/178178405-fe853845-cd5d-4c8f-a679-1d2592ae18b5.gif)
# RL Agent Orchestration Pipeline

### Reinforcement Learning | SARSA | Q-Learning | Cliff Walking

A hands-on implementation of Reinforcement Learning agents using **SARSA and Q-Learning** in the Cliff Walking environment. This project explores agent-environment interaction, epsilon-greedy action selection, temporal-difference learning, Q-table updates, and policy evaluation.

---

## Overview

Reinforcement Learning (RL) is a machine learning paradigm in which an agent learns to make decisions by interacting with an environment and receiving rewards or penalties.

This project implements and explores multiple agent strategies in the classic **Cliff Walking** environment.

The primary objective is to understand how different learning strategies enable an agent to discover effective policies while navigating an environment with negative rewards and terminal states.

## Key Components

- **Random Agent:** Establishes a baseline through random action selection.
- **SARSA Agent:** Implements on-policy Temporal-Difference control.
- **Q-Learning Agent:** Implements off-policy Temporal-Difference control.
- **Q-Table Storage:** Saves learned state-action values using Python Pickle.
- **Agent Evaluation:** Loads a trained Q-table and executes the learned policy.
- **Environment Visualization:** Uses OpenCV to visualize the agent navigating the Cliff Walking grid.

## Environment: Cliff Walking

The project uses the `CliffWalking-v0` environment from OpenAI Gym.

The environment consists of a 4 × 12 grid in which an agent must navigate from a starting position to a goal while avoiding the cliff.

| Environment Property | Description |
|---|---|
| Environment | Cliff Walking |
| State Space | 48 discrete states |
| Action Space | 4 actions |
| Actions | Up, Down, Left, Right |
| Learning Approach | Tabular Reinforcement Learning |
| Reward Mechanism | Positive and negative rewards |
| Objective | Reach the goal while avoiding the cliff |

## Agent Learning Algorithms

### 1. SARSA — On-Policy TD Control

SARSA updates its action-value estimates using the action actually selected by the current policy.

**Update Equation:**

```text
Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
```

Where:

- S: Current state
- A: Current action
- R: Immediate reward
- S': Next state
- A': Next action selected by the policy
- α: Learning rate
- γ: Discount factor

**Implementation:** `sarsa.py`

### 2. Q-Learning — Off-Policy TD Control

Q-Learning updates its action-value estimates using the maximum estimated value of the next state.

**Update Equation:**

```text
Q(S,A) ← Q(S,A) + α[R + γ max Q(S',a) - Q(S,A)]
```

Where:

- S: Current state
- A: Current action
- R: Immediate reward
- S': Next state
- a: Possible next action
- α: Learning rate
- γ: Discount factor

**Implementation:** `q_learning.py`

### 3. Random Agent

A baseline agent that selects actions randomly without learning an optimized policy.

**Implementation:** `random_agent.py`

## Project Architecture

```text
RL-Agent-Orchestration-Pipeline/
│
├── cv_show.py
├── evaluator.py
├── q_learning.py
├── q_learning_q_table.pkl
├── random_agent.py
├── requirements.txt
├── sarsa.py
├── sarsa_q_table.pkl
└── README.md
```

## Agent Execution Pipeline

```text
Environment Initialization
          |
          v
    Agent Initialization
          |
          v
   State Observation
          |
          v
    Action Selection
          |
          v
 Environment Interaction
          |
          v
   Reward & Next State
          |
          v
   Q-Value Update
          |
          v
   Policy Improvement
          |
          v
    Training Complete
          |
          v
    Q-Table Storage
          |
          v
   Agent Evaluation
```

## Hyperparameters

The training implementations use the following parameters:

| Parameter | Value |
|---|---:|
| Learning Rate (α) | 0.1 |
| Discount Factor (γ) | 0.9 |
| Exploration Rate (ε) | 0.1 |
| Training Episodes | 500 |

These values are defined in the training scripts.

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Pranav123221/RL-Agent-Orchestration-Pipeline.git
```

### 2. Navigate to the project directory

```bash
cd RL-Agent-Orchestration-Pipeline
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run SARSA training

```bash
python sarsa.py
```

### 5. Run Q-Learning training

```bash
python q_learning.py
```

### 6. Run the Random Agent

```bash
python random_agent.py
```

### 7. Evaluate the trained Q-Learning agent

```bash
python evaluator.py
```

### 8. Visualize the Cliff Walking environment

```bash
python cv_show.py
```

**Note:** The project uses legacy Gym APIs and pinned dependencies. Compatibility with newer Python, Gymnasium, or OpenCV versions may require code updates. The visualization scripts require a graphical display.

## Technologies & Libraries

- Python
- OpenAI Gym
- NumPy
- OpenCV
- Pickle

## Learning Objectives

Through this implementation, the project explores:

- Agent-environment interaction
- State and action spaces
- Reward-based learning
- Exploration vs. exploitation
- Epsilon-greedy action selection
- Temporal-Difference learning
- On-policy vs. off-policy learning
- State-action value estimation
- Q-table persistence
- Policy execution and evaluation

## Future Improvements

- Modernize the environment implementation using Gymnasium.
- Add training reward and episode-length analytics.
- Compare SARSA and Q-Learning performance.
- Build an interactive agent visualization dashboard.
- Introduce configurable training parameters.
- Extend the pipeline with additional RL environments.

## Learning & Attribution

This repository is maintained as a Reinforcement Learning learning project and educational adaptation.

The initial implementation is based on the Reinforcement Learning tutorial series by [Rajtilak LS](https://github.com/rajtilakls2510/reinforcement_learning), specifically its Part 3 Cliff Walking implementation.

Original Repository: https://github.com/rajtilakls2510/reinforcement_learning

Original author attribution and applicable licensing terms should be preserved.

---

**Author:** Pranav Sharma

**GitHub:** [@Pranav123221](https://github.com/Pranav123221)

**Project:** RL-Agent-Orchestration-Pipeline
