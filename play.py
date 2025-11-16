import argparse
import time
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformObservation, RecordVideo, FrameStackOberservation as FrameStack
from stable_baselines3 import DQN


def make_eval_env(env_id: str,
                  render_mode: str | None = "human",
                  frame_stack: int = 4,
                  grayscale: bool = True) -> gym.Env:

    env = gym.make(env_id, render_mode=render_mode)
    # Typical DQN Atari preprocessing:
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=grayscale,
        scale_obs=False,     # SB3’s DQN expects uint8 images by default
        screen_size=84
    )
    env = FrameStack(env, frame_stack)
    return env


def maybe_transpose_to_match_model(env: gym.Env, model_obs_shape: tuple | None):

    e_shape = getattr(env.observation_space, "shape", None)
    if model_obs_shape is None or e_shape is None or e_shape == model_obs_shape:
        return env, False

    if len(e_shape) == 3 and len(model_obs_shape) == 3:
        # Are they the same dims in a different order? (common case)
        if sorted(e_shape) == sorted(model_obs_shape):
            # Most common: env HWC -> model CHW
            if model_obs_shape[0] == e_shape[-1] and model_obs_shape[1] == e_shape[0] and model_obs_shape[2] == e_shape[1]:
                def _to_chw(obs):
                    # FrameStack returns LazyFrames; ensure np.array then transpose.
                    return np.transpose(np.array(obs), (2, 0, 1))
                env = TransformObservation(env, _to_chw)
                return env, True

    # If we get here, shapes are not a simple transpose; we leave as-is.
    return env, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5",
                        help="Atari env id (MUST match what you used in training).")
    parser.add_argument("--model-path", type=str, default="dqn_model.zip",
                        help="Path to the saved model from train.py")
    parser.add_argument("--episodes", type=int, default=5,
                        help="How many episodes to run.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for env.reset() to make runs reproducible.")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable GUI window (useful for servers).")
    parser.add_argument("--record-video", type=str, default=None,
                        help="Directory to save a video. "
                             "Note: Gymnasium supports *one* render_mode. "
                             "If you set this, we use 'rgb_array' when --no-render is also used.")
    parser.add_argument("--fps", type=int, default=60,
                        help="Approximate display FPS cap when rendering; <=0 runs as fast as possible.")
    args = parser.parse_args()

    if args.record_video and args.no_render:
        render_mode = "rgb_array"
    elif args.no_render:
        render_mode = None
    else:
        render_mode = "human"

    # Build env and load model
    env = make_eval_env(args.env_id, render_mode=render_mode)
    model = DQN.load(args.model_path, device="auto")

    # Try to align the observation layout with what the model expects
    model_obs_shape = getattr(model, "observation_space", None)
    model_obs_shape = model_obs_shape.shape if model_obs_shape is not None else None
    env, transposed = maybe_transpose_to_match_model(env, model_obs_shape)

    # Optional: record a video (only when render_mode supports arrays)
    if args.record_video:
        if render_mode != "rgb_array":
            print("[WARN] Recording video requires render_mode='rgb_array'. "
                  "Run with --no-render --record-video path/to/dir to record without GUI.")
        else:
            env = RecordVideo(env, video_folder=args.record_video,
                              name_prefix="dqn_play", episode_trigger=lambda ep: ep == 0)
            print(f"[INFO] Will record episode 1 to: {args.record_video}")

    print(f"[INFO] Loaded: {args.model_path}")
    print(f"[INFO] Env: {args.env_id} | Render: {render_mode or 'off'} | Episodes: {args.episodes}")
    if transposed:
        print("[INFO] Applied HWC -> CHW transpose to match the model (C, H, W).")

    # Run evaluation with greedy policy (deterministic=True)
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed)
        ep_reward, ep_len = 0.0, 0
        done = False
        t0 = time.perf_counter()

        while not done:
            action, _ = model.predict(obs, deterministic=True)  # GreedyQPolicy equivalent
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            ep_reward += float(reward)
            ep_len += 1

            if render_mode == "human":
                env.render()
                if args.fps and args.fps > 0:
                    time.sleep(1.0 / args.fps)

        dt = time.perf_counter() - t0
        print(f"Episode {ep+1}/{args.episodes} | Return: {ep_reward:.1f} | "
              f"Length: {ep_len} steps | {dt:.1f}s")

    env.close()


if __name__ == "__main__":
    main()