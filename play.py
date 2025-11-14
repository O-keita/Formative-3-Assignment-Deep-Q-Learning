import os
os.environ["DISPLAY"] = ":0"
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
import cv2
import numpy as np
from collections import deque

# --- Configuration ---
ENV_ID = "ALE/Riverraid-v5"
MODEL_PATH = "checkpoints/CnnPolicy/best_model/best_model.zip"

# --- Helper: preprocess frame like SB3 training ---
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame

# --- Play the model visually ---
def play_model():
    env = gym.make(ENV_ID, render_mode="human")  # opens a game window
    model = DQN.load(MODEL_PATH)

    obs, info = env.reset()
    frame_stack = deque(maxlen=4)
    f = preprocess_frame(obs)
    for _ in range(4):
        frame_stack.append(f)

    done = False
    total_reward = 0

    while not done:
        stacked_obs = np.stack(frame_stack, axis=0)
        stacked_obs = np.expand_dims(stacked_obs, axis=0)  # add batch dimension

        action, _ = model.predict(stacked_obs, deterministic=True)
        action = int(action.item())  # convert numpy array to scalar safely
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        f = preprocess_frame(obs)
        frame_stack.append(f)

        env.render()  # show frame each step


    print("✅ Total reward:", total_reward)
    env.close()

if __name__ == "__main__":
    play_model()
