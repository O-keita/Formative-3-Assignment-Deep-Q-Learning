# Formative-3-Assignment-Deep-Q-Learning

This project implements a Deep Q-Learning (DQN) agent to play the Atari game Riverraid using the Stable Baselines3 library and Gymnasium. It serves as a formative assignment for the Machine Learning course, demonstrating reinforcement learning techniques on classic Atari environments.

## Environment Details

- **Game** : River Raid (ALE/Riverraid-v5)
- **Action Space:** : Discrete(18)
- **Observation Space** :  (210, 160, 3) RGB image
- **Objective** : Control a jet to destroy enemy objects while avoiding collisions and managing fuel

  ### Scoring System
| Enemy Object | Score Points |
|--------------|--------------|
| Tanker       | 30           |
| Helicopter   | 60           |
| Fuel Depot   | 80           |
| Jet          | 100          |
| Bridge       | 500          |

   

## Project Structure

- `train.py`: Main training script for the DQN agent
- `play.py`: Evaluation script to play/test trained models
- `initial_env.py`: Simple script to visualize the Riverraid environment
- `requirements.txt`: Python dependencies
- `.gitignore`: Git ignore file for virtual environments, checkpoints, and logs

## Features

- **Training**: Train a DQN agent on Atari Riverraid with configurable hyperparameters
- **Evaluation**: Play trained models with rendering, video recording, and performance metrics
- **Checkpoints**: Automatic model saving during training
- **Logging**: TensorBoard integration for monitoring training progress
- **Frame Stacking**: Uses 4-frame stacking for better temporal understanding

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/O-keita/Formative-3-Assignment-Deep-Q-Learning.git
   cd Formative-3-Assignment-Deep-Q-Learning
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `gymnasium[atari]`: Atari environments
   - `ale-py`: Arcade Learning Environment
   - `autorom`: Automatic ROM installation
   - `stable-baselines3[extra]`: DQN implementation with extra features
   - `gymnasium[accept-rom-license]`: ROM license acceptance
   - `moviepy`: Video recording for gameplay demonstrations

4. Install Atari ROMs:
   ```bash
   AutoROM --accept-license
   ```
   This downloads and installs the Atari 2600 ROMs required for the games.

## Usage

### Training the Agent

Run the training script to train a DQN agent on Riverraid:

```bash
python train.py --env ALE/Riverraid-v5 --policy cnn --timesteps 100000 --lr 0.0003 --gamma 0.99 --batch_size 32 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_fraction 0.1
```

You can customize hyperparameters via command-line arguments. Default values are set for Riverraid.

This will:
- Train for the specified timesteps using 4 parallel environments
- Save checkpoints every 10,000 steps in the `checkpoints/` directory
- Log training progress to TensorBoard in the `logs/` directory
- Evaluate the model every 1,000 steps
- Save the final model as `checkpoints/{policy}/dqn_model_final.zip`

### Playing/Evaluating the Trained Model

After training, evaluate the model:

```bash
python play.py --model-path checkpoints/CnnPolicy/dqn_model_final.zip
```

Additional options:
- `--env-id`: Specify Atari environment (default: ALE/Breakout-v5, but trained on Riverraid)
- `--episodes`: Number of episodes to play (default: 5)
- `--no-render`: Disable GUI rendering
- `--record-video`: Record gameplay to a directory
- `--fps`: Control rendering speed

Example with video recording:
```bash
python play.py --model-path checkpoints/CnnPolicy/dqn_model_final.zip --record-video ./videos --no-render
```

### Advanced Usage Examples

Here are various command examples for different scenarios:

**Play with GUI rendering (opens game window):**
```bash
python play.py --env-id ALE/Riverraid-v5 --model-path checkpoints/CnnPolicy/dqn_model_final.zip
```

**Play without rendering (headless/server environments):**
```bash
python play.py --env-id ALE/Riverraid-v5 --model-path checkpoints/CnnPolicy/dqn_model_final.zip --no-render --episodes 1
```

**Play multiple episodes:**
```bash
python play.py --env-id ALE/Riverraid-v5 --model-path checkpoints/CnnPolicy/dqn_model_final.zip --no-render --episodes 3
```

**Record gameplay video:**
```bash
python play.py --env-id ALE/Riverraid-v5 --model-path checkpoints/CnnPolicy/dqn_model_final.zip --record-video ./videos --no-render --episodes 1
```

**Note:** All commands should be run from the activated virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Then run any of the above commands
```

### Visualizing the Environment

To see a random agent playing Riverraid:

```bash
python initial_env.py
```

## Configuration

The script accepts command-line arguments to customize hyperparameters. Run `python train.py --help` for details.

Default hyperparameters:
- `env`: "ALE/Riverraid-v5"
- `timesteps`: 300,000
- `lr`: 0.0003
- `gamma`: 0.79
- `batch_size`: 128
- `epsilon_start`: 1.0
- `epsilon_end`: 0.02
- `epsilon_fraction`: 0.5

Use command-line arguments to experiment with different configurations without editing the code.

## Requirements

- Python 3.8+
- See `requirements.txt` for specific package versions


## Hyperparameter Experiments
   ### Omar Keita
   
   | Exp | Time Steps | Learning Rate (lr) | Gamma (γ) | Batch Size | ε_start | ε_end | ε_decay | Observed Behavior |
   |-----|------------|--------------------|-----------|-------------|----------|--------|----------|--------------------|
   | 1 | 100,000 | 0.00001 | 0.99 | 32 | default | 0.01 | 0.1 | Mean reward **4.7 ± 2.62**, episode length **29.1 ± 13.8**. Agent dies early, barely survives initial obstacles. Needs more training and tuning. |
   | 2 | 100,000 | 0.0005 | 0.99 | 64 | 1.0 | 0.01 | 0.2 | Mean reward **7.32**, noticeably better than Exp 1. Agent survives slightly longer and begins learning basic movement patterns. |
   | 3 | 250,000 | 0.0005 | 0.99 | 64 | 1.0 | 0.02 | 0.4 | Mean reward **8.90 ± 6.65**. Strong early exploration, more exploitation after ~100k steps. Avg episode length ~33 steps. Completed **9728 episodes**. Shows gradual improvement. |
   | 4 | 250,000 | 0.0003 | 0.99 | 64 | 1.0 | 0.03 | 0.5 | Mean reward **9.28 ± 7.46**. More balanced exploration → better stability. Avg episode length ~38 steps. Completed **9004 episodes**. Loss observed ~0.115. |
   | 5 | 250,000 | 0.00003 | 0.79 | 128 | 1.0 | 0.002 | 0.3 | Mean reward dropped — likely due to **lower gamma** and **very small decay**. But completed **9628 episodes**, showing higher turnover. Needs higher γ and slower decay. |
   | 6 | 300,000 | 0.0003 | 0.79 | 128 | 1.0 | 0.02 | 0.5 | The mean reward improved to 9.4, and the agent completed 11,252 episodes. This reflects clear progress, performance became more stable, and the agent survived longer before termination. However, further tuning is still needed to achieve stronger and more consistent gameplay. |

   ### Jean Pierre  Niyongabo
   | Exp | Time Steps | Learning Rate (lr) | Gamma (γ) | Batch Size | ε_start | ε_end | ε_decay | Observed Behavior |
   |-----|------------|--------------------|-----------|-------------|----------|--------|----------|--------------------|
   | 1 | 50,000 | 0.001 | 0.95 | 32 | 1.0 | 0.05 | 0.2 | Mean reward stabilized at ~2.1, episode length ~21 steps. Agent learned basic survival patterns but performance plateaued. Completed ~2244 episodes. Training stable but limited improvement in later stages. |
   | 2 | 100,000 | 0.0005 | 0.99 | 64 | 1.0 | 0.01 | 0.3 | Significant improvement over Exp 1. Mean reward reached ~3.8, episode length ~21 steps. Completed ~5080 episodes. Much more stable training with lower loss values. Best evaluation reward of 4.48. Agent learned better survival and scoring patterns. |
   | 3 | 150,000 | 0.0001 | 0.90 | 128 | 1.0 | 0.02 | 0.4 | Training crashed before completion. At 101,788 steps, eval mean reward 4.94, rollout 5.58. |
   | 4 | 200,000 | 0.00005 | 0.85 | 32 | 1.0 | 0.08 | 0.1 | Mean reward 6.32 ± 4.11, episode length 24.18 ± 11.46. Completed 10,132 episodes. Best eval 7.22 ± 5.25 at 196k steps. Agent showed stable performance with moderate rewards, better than exp 3 but lower than exp 5. |
   | 5 | 250,000 | 0.0008 | 0.98 | 64 | 1.0 | 0.03 | 0.5 | Mean reward 7.86 ± 6.34, episode length ~32. Best eval 8.62 ± 6.35. Completed 10,640 episodes. Improved performance over exp 4 with higher rewards and longer episodes, showing better learning stability. |
   | 6 | 300,000 | 0.0003 | 0.92 | 128 | 1.0 | 0.06 | 0.25 |  |
   | 7 | 350,000 | 0.00001 | 0.88 | 32 | 1.0 | 0.04 | 0.35 |  |
   | 8 | 400,000 | 0.0007 | 0.96 | 64 | 1.0 | 0.07 | 0.15 |  |
   | 9 | 450,000 | 0.0002 | 0.94 | 128 | 1.0 | 0.09 | 0.45 |  |
   | 10 | 500,000 | 0.0009 | 0.97 | 32 | 1.0 | 0.01 | 0.3 |  |


   ### Lievin Murayire
   | Exp | Time Steps | Learning Rate (lr) | Gamma (γ) | Batch Size | ε_start | ε_end | ε_decay | Observed Behavior |
   |-----|------------|--------------------|-----------|-------------|----------|--------|----------|--------------------|


   ### Jean Paul Irakoze
   | Exp | Time Steps | Learning Rate (lr) | Gamma (γ) | Batch Size | ε_start | ε_end | ε_decay | Observed Behavior |
   |-----|------------|--------------------|-----------|-------------|----------|--------|----------|--------------------|
   
## Demo Video
**Agent Playing River Raid**


https://github.com/user-attachments/assets/2c0e1c3d-f644-4dff-865c-8a86bc5b38bc




## Notes

- Training may take several minutes to hours depending on your hardware
- The project uses CNN policy suitable for image-based Atari games
- Checkpoints and logs are ignored by git to keep the repository clean
- For best results, ensure you have a GPU available for faster training

## License

This project is part of an educational assignment. Please refer to the course guidelines for usage permissions.
