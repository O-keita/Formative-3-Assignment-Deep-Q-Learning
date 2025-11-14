import gymnasium as gym
import ale_py


def show_env(env_id):
    env = gym.make(env_id, render_mode="human")
    obs, info = env.reset()
    
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()


if __name__ == "__main__":

    show_env("ALE/Riverraid-v5")
