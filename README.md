# Reinforcement-Learning-Model-for-Heat-Diffusion-Reaction-Diffusion-with-Surrogate-Environment
Optimizing aluminum-oxygen combustion processes through physics-based simulation, surrogate modeling, and reinforcement learning control

Project Overview

This project develops an intelligent control system for aluminum combustion reactors using a novel combination of:

Physics-based PDE modeling of reaction-diffusion and heat transfer
GPU-accelerated simulation reducing computation time from hours to seconds
Hybrid ML surrogate models for real-time state prediction
Actor-Critic reinforcement learning for optimal process control

Key Achievements

🚀 2000x speed improvement through CUDA GPU parallelization
🎯 R² = 0.9975 temperature prediction accuracy
⚡ Real-time control capability with millisecond response times
🏆 Energy-efficient optimization balancing combustion completeness with operational costs

├── src/                          # Source code (branch: src)

│   ├── gpu_batch_simulation.py   # CUDA-accelerated batch simulations

│   ├── cpu_batch_simulation.py   # CPU-based simulation baseline

│   ├── surrogate_model.py        # Hybrid ML model (Gradient Boosting + Neural Network)

│   └── rl_model.py               # Actor-Critic PPO implementation

├── data/                         # Results and visualizations (branch: data)

│   ├── heatmaps/                 # Temperature and concentration distributions

│   ├── training_curves/          # RL convergence plots

│   └── reward_landscapes/        # Policy optimization visualizations

└── documentation/                # Detailed technical report (branch: documentation)

    └── technical_report.pdf      # Complete project documentation

# Required packages
pip install numpy scipy matplotlib torch scikit-learn
pip install stable-baselines3 gym

# For GPU acceleration (optional but recommended)
# Install CUDA toolkit and cupy
pip install cupy-cuda11x  # or appropriate CUDA version

# GPU-accelerated (recommended)
python src/gpu_batch_simulation.py

# CPU baseline (slower)
python src/cpu_batch_simulation.py
python src/surrogate_model.py
python src/rl_model.py
