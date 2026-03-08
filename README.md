# Slither: Deep Q-Learning Snake

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-orange)
![Pygame](https://img.shields.io/badge/UI-Pygame-green)
![Status](https://img.shields.io/badge/Project-Portfolio%20Ready-brightgreen)

Slither is a reinforcement learning portfolio project where a Snake agent learns to survive and grow inside a grid world.
The project combines game simulation, feature engineering, neural-network training, and real-time visualization in a compact codebase.

## Demo

Project walkthrough video: [YouTube Demo](https://youtu.be/tioUIGuw5b4)

## Why This Project Exists

This repo demonstrates end-to-end ML engineering skills in a small but complete system:

- Environment design with deterministic game rules and reward signals.
- Feature extraction from raw board state into model-ready vectors.
- Deep Q-learning training loop with replay-style updates.
- CLI-based workflows for training, exhibition, log replay, and manual play.

## Quick Start

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train a model
python main.py -sessions 100

# 4) Watch the trained agent
python main.py -exhibit
```

## Usage

### Main modes

```bash
# Human mode
python main.py -play

# Train from live simulated episodes
python main.py -sessions 200 -save models/latest.keras

# Exhibit a saved model (greedy policy)
python main.py -exhibit -load models/latest.keras

# Train from an existing JSONL gameplay log
python main.py -logtrain -logfile logs/snakelog.jsonl -load models/latest.keras -save models/latest.keras
```

### CLI flags

- `-play`: human-controlled game.
- `-exhibit`: run agent in visual mode with epsilon set to zero.
- `-logtrain`: train from historical logs.
- `-sessions N`: number of episodes for train/play/exhibit loops.
- `-load PATH`: model to load (default `models/latest.keras`).
- `-save PATH`: model destination (default `models/latest.keras`).
- `-logfile PATH`: replay log file path (default `logs/snakelog.jsonl`).
- `-debug`: pause each step for inspection.

## System Architecture

```text
main.py
  -> Bill.py (orchestrator)
      -> Game.py (environment + rendering)
      -> Interpreter.py (state -> features + reward mapping)
      -> Agent.py (DQN model + action selection + updates)
```

## Reinforcement Learning Setup

### State representation

The agent does not consume the raw board directly. Instead, `Interpreter.py`:

- Extracts directional views from the snake head (`north`, `south`, `east`, `west`).
- Encodes distances, inverse distances, blocked paths, and object counts.
- Produces `nfeatures = 78` engineered inputs (configured in `game_settings.py`).

### Action space

- `0`: down
- `1`: up
- `2`: right
- `3`: left

### Reward function

Configured in `game_settings.py`:

- Empty move (`0`): `-10`
- Wall collision (`W`): `-1000`
- Self collision (`S`): `-1000`
- Red apple (`R`): `-500`
- Green apple (`G`): `+500`

### Learning policy

- Epsilon-greedy exploration during training.
- Q-value updates with discount factor `gamma`.
- Learning-rate-style update scaling with `alpha`.
- Neural network implemented with TensorFlow/Keras dense layers.

## Repository Map

```text
.
├── main.py                 # CLI entry point
├── Bill.py                 # high-level runtime orchestration
├── Game.py                 # snake environment + pygame rendering
├── Interpreter.py          # feature engineering + log utilities
├── Agent.py                # model definition, action policy, training
├── game_settings.py        # rewards, hyperparameters, board settings
├── models/                 # saved checkpoints
└── README.md
```

## Reproducibility Notes

- GPU is explicitly disabled in `main.py` (`CUDA_VISIBLE_DEVICES=-1`) for CPU-first portability.
- Hyperparameters and rewards are centralized in `game_settings.py`.
- Several pretrained checkpoints are included under `models/`.

## Current Limitations

- No fixed random seed across all components, so runs are not fully deterministic.
- No automated benchmark script yet for standardized evaluation metrics.
- No test suite/CI pipeline included yet.
- Training diagnostics (plots/tables) are not persisted by default.

## Roadmap

- Add a headless evaluation script with metrics (mean score, max length, survival steps).
- Add test coverage for environment mechanics and feature extraction.
- Add CI checks (lint + smoke tests).
- Compare baseline DQN with Double DQN / target network variants.

## Portfolio Summary

Slither is intentionally scoped as a practical ML engineering project: small enough to review quickly, but complete enough to show environment design, model training logic, and decision-making tradeoffs in one repo.
