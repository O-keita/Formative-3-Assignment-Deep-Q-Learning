import gymnasium as gym
import ale_py

env = gym.make("ALE/Riverraid-v5", render_mode="human")

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()