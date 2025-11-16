import os

import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

#========================================================
# Config / Hyperparameters
#========================================================
ENV_ID = "ALE/Riverraid-v5"
N_ENVS = 4
N_STACK = 4
SEED = 42
TIMESTEPS = 1_000

# Hyperparameters for this run
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99
EXPLORATION_FRACTION = 0.1

CHECKPOINT_FREQ = 5_000
EVAL_FREQ = 1_000
N_EVAL_EPISODES = 50  

#========================================================
# Environment Setup Function
#========================================================
def make_env(env_id, n_envs=N_ENVS, n_stack=N_STACK, seed=SEED):
    """
    A vectorized Atari environment with frame stacking and monitoring.
    """
    env = make_atari_env(env_id, n_envs=n_envs, seed=seed)
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=n_stack)
    print(f"Environment '{env_id}' ready: {n_envs} envs, {n_stack}-frame stack")
    return env

#========================================================
# Training function
#========================================================
def train_dqn(policy_name="CnnPolicy", env_id=ENV_ID):

    # Prepare environments
    train_env = make_env(env_id)
    eval_env = make_env(env_id, n_envs=1)

    # Model setup
    model = DQN(
        policy=policy_name,
        env=train_env,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        buffer_size=20_000,
        learning_starts=30_000,
        target_update_interval=5000,
        train_freq=4,
        gamma=GAMMA,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="logs/"
    )
# can increase for more stable evaluation
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path="./checkpoints/",   # <-- folder where checkpoints are stored
        name_prefix="dqn_model"
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"checkpoints/{policy_name}/best_model/",
        log_path=f"logs/{policy_name}/eval/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    # Train the model
    print(f"\n=== Training {policy_name} with lr={LEARNING_RATE}, gamma={GAMMA}, batch={BATCH_SIZE} ===")
    callback_list = CallbackList([checkpoint_cb, eval_cb])
    model.learn(total_timesteps=TIMESTEPS, callback=callback_list)

    # Save final model
    final_model_path = f"checkpoints/{policy_name}/dqn_model_final.zip"
    model.save(final_model_path)
    print(f"Saved final model at: {final_model_path}")

    # Evaluate final model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES)
    print(f"Final evaluation: mean_reward={mean_reward:.2f} ± {std_reward:.2f}\n")

    # Close environments
    train_env.close()
    eval_env.close()

#========================================================
# Main function
#========================================================
def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Change the policy manually to "CnnPolicy" or "MlpPolicy" before each run
    train_dqn(policy_name="CnnPolicy")

if __name__ == "__main__":
    main()
