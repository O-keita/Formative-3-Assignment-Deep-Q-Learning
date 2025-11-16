# Formative-3-Assignment-Deep-Q-Learning

This project implements a Deep Q-Learning (DQN) agent to play the Atari game Riverraid using the Stable Baselines3 library and Gymnasium. It serves as a formative assignment for the Machine Learning course, demonstrating reinforcement learning techniques on classic Atari environments.

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
   git clone <repository-url>
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

## Usage

### Training the Agent

Run the training script to train a DQN agent on Riverraid:

```bash
python train.py
```

This will:
- Train for 100,000 timesteps using 4 parallel environments
- Save checkpoints every 5,000 steps in the `checkpoints/` directory
- Log training progress to TensorBoard in the `logs/` directory
- Evaluate the model every 1,000 steps
- Save the final model as `checkpoints/CnnPolicy/dqn_model_final.zip`

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

### Visualizing the Environment

To see a random agent playing Riverraid:

```bash
python initial_env.py
```

## Configuration

Key hyperparameters in `train.py`:
- `ENV_ID`: "ALE/Riverraid-v5"
- `TIMESTEPS`: 100,000
- `LEARNING_RATE`: 1e-4
- `BATCH_SIZE`: 32
- `GAMMA`: 0.99
- `EXPLORATION_FRACTION`: 0.1

Modify these values in `train.py` before training to experiment with different configurations.

## Requirements

- Python 3.8+
- See `requirements.txt` for specific package versions

## Notes

- Training may take several minutes to hours depending on your hardware
- The project uses CNN policy suitable for image-based Atari games
- Checkpoints and logs are ignored by git to keep the repository clean
- For best results, ensure you have a GPU available for faster training

## License

This project is part of an educational assignment. Please refer to the course guidelines for usage permissions.