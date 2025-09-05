# Slither: A Reinforcement Learning Snake Adventure

![Snake Banner](https://img.shields.io/badge/AI-Snake-brightgreen)

Welcome to **Slither**, my portfolio project that turns the classic Snake game into a playground for reinforcement learning. It’s built to be eye‑catchy for recruiters, but it still doubles as a solid starting point for anyone curious about teaching an AI to survive in a grid full of food and danger.

## How it Works
- **Bill the Snake Charmer** orchestrates everything: training sessions, exhibitions, and human‑playable runs.
- **Agent** is a deep Q‑learning model built with TensorFlow and trained via replayed game logs.
- **Interpreter** translates raw game history into feature vectors and rewards, feeding the neural network with memories of past moves.
- **Game** wraps a Pygame interface where the snake slithers, eats, and inevitably learns from failure.

Run it like this:
```bash
# Train from scratch
python main.py -sessions 100

# Show off the current model
python main.py -exhibit

# Play yourself
python main.py -play
```

## Reinforcement Learning in a Nutshell
Slither uses **Deep Q‑Learning**, a technique where the agent learns a value function \(Q(s, a)\) predicting the expected return of taking action \(a\) in state \(s\). The update rule is:
\[
Q_{new} \leftarrow Q_{old} + \alpha \big( r + \gamma \max_{a'} Q(s', a') - Q_{old} \big)
\]
- **\(\alpha\)** – learning rate, how fast we update.
- **\(\gamma\)** – discount factor, how much future rewards matter.
- **Epsilon‑greedy** action selection balances exploration and exploitation so the snake doesn't loop forever.

## Roadmap & Video
I’m preparing a short video walkthrough of the project. Stay tuned: [insert link]

## Installation
```bash
pip install -r requirements.txt
```

## Portfolio Note
This project is designed to showcase my ability to combine machine learning theory with playful, interactive demos. Recruiters, feel free to dive into the code or just enjoy the snake’s dance.

---
*Hungry for feedback. Let me know what you think!*

