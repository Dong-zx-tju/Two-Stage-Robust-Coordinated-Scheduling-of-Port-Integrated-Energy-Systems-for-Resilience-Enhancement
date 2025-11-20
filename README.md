# Dynamic Fault Reconfiguration in Ship Power Systems (DQN-MASK)

This repository provides a lightweight reference implementation of the
DQN-MASK approach from _Dynamic Fault Reconfiguration of Distribution Networks
in Ship Power Systems Based on Deep Reinforcement Learning_ (IEEE TTE, 2024).
The code builds a simplified medium-voltage DC shipboard power network and
trains a Deep Q-Network with action masking to recover service after random
line faults.

## Project layout
- `sps_reconfig/environment.py` – small RL environment that models zonal loads,
  generators, line capacities, and switchable breakers. The environment exposes
  a legal-action mask (with a built-in no-op action) that prevents islanding or
  thermal overloads.
- `sps_reconfig/dqn_mask.py` – PyTorch implementation of the masked DQN agent,
  including replay buffer, epsilon-greedy exploration, target network updates,
  and a helper function to run training.
- `train.py` – command-line entry point that trains the agent and saves a JSON
  file with the episode returns.
- `requirements.txt` – Python dependencies.

## Quick start
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run a short training session (300 episodes by default):
   ```bash
   python train.py --episodes 300 --seed 0 --save artifacts
   ```
   The script prints the average return over the last 10 episodes and writes
   `artifacts/training_curve.json` so you can plot learning curves.

## Notes on the simplifications
The simulation is intentionally compact to keep CPU runtime low. It captures the
key ideas from the paper—weighted load restoration, line capacity checks, and
masking of unsafe switching actions—while abstracting away detailed power-flow
physics. You can extend the `canonical_graph()` definition to match other test
systems or import the environment into your own research scripts.
