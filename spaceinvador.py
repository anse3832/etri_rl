import os
import time
import gymnasium as gym
import torch

from sb3_contrib import QRDQN
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import numpy as np

from callback import SaveOnBestTrainingRewardCallback


############################################################
# Atari Wrapper (Gymnasium 최신 방식)
############################################################
def make_atari_env(env_id, log_dir=None, render_mode=None):
    # Gymnasium Atari 환경은 ALE/xxx 형식을 사용
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)

    # === DQN 표준 Atari preprocessing ===
    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        frame_skip=4,
        screen_size=84,
        scale_obs=False
    )

    # === FrameStack ===
    env = FrameStack(env, 4)

    # === Logging ===
    if log_dir is not None:
        env = Monitor(env, log_dir)

    return env


############################################################
# TRAIN
############################################################
def train(
    env_id="ALE/SpaceInvaders-v5",
    model_type='DQN',
    total_timesteps=2_000_000,
    log_base_dir="logs",
    model_base_dir="models"
):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, log_base_dir, env_id.replace("/", "_"))
    model_dir = os.path.join(script_dir, model_base_dir)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=== Building Atari Environment ===")
    env = make_atari_env(env_id, log_dir=log_dir)

    if model_type == "DQN":
        model_path = os.path.join(model_dir, f"{env_id.replace('/', '_')}_DQN")
    elif model_type == "QRDQN":
        model_path = os.path.join(model_dir, f"{env_id.replace('/', '_')}_QRDQN")

    # Atari 기본 하이퍼파라미터 (출처: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml)
    if model_type == "DQN":
        print("=== Building DQN + CNNPolicy ===")
        model = DQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=100_000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            tensorboard_log=log_dir,
            verbose=1,
            device="cuda"
        )
    elif model_type == "QRDQN":
        print("=== Building QRDQN + CNNPolicy ===")
        model = QRDQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=6.25e-5,
            buffer_size=500_000,
            learning_starts=80_000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=32_000,
            exploration_fraction=0.5,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            policy_kwargs=dict(
                n_quantiles=200,
                net_arch=[512],
            ),
            tensorboard_log=log_dir,
            verbose=1,
            device="cuda"
        )

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=10000,
        log_dir=log_dir
    )

    print("=== Training Started ===")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    print("=== Training Finished ===")

    model.save(model_path)
    print(f"Model saved: {model_path}")

    env.close()
    return model_path


############################################################
# RUN
############################################################
def run(env_id="ALE/SpaceInvaders-v5", model_type="DQN", model_path=None, n_episodes=10):

    if model_path is None:
        raise ValueError("model_path must be provided")

    print("=== Running trained model ===")

    # ← 학습 환경과 동일한 preprocessing + FrameStack 적용!!
    env = make_atari_env(env_id, render_mode="human")

    if model_type == "DQN":
        model = DQN.load(model_path, env)
    elif model_type == "QRDQN":
        model = QRDQN.load(model_path, env)

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            # LazyFrames → numpy 변환
            if hasattr(obs, "_frames"):     # SB3의 FrameStack 형태
                obs = np.array(obs._frames)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            time.sleep(0.01)
            done = terminated or truncated

        print(f"[Episode {ep+1}] Reward = {total_reward}")

    env.close()



############################################################
# MAIN
############################################################
if __name__ == "__main__":
    ENV_ID = "ALE/SpaceInvaders-v5"

    # training code
    # model_type = "QRDQN" # "DQN"
    # if model_type == "DQN":
    #     print("=== Training SpaceInvaders with DQN + CNNPolicy ===")
    #     model_path = train(ENV_ID, model_type="DQN")
    # elif model_type == "QRDQN":
    #     print("=== Training SpaceInvaders with QRDQN + CNNPolicy ===")
    #     model_path = train(ENV_ID, model_type="QRDQN", total_timesteps=10_000_000)

    # model_path = "./models/ALE_SpaceInvaders-v5/best_model/best_model_DQN.zip"
    # print("=== Running best DQN model ===")
    # run(ENV_ID, model_type="DQN", model_path=model_path, n_episodes=10)

    # test code (best QRDQN model)
    model_path = "./best_model_QRDQN.zip"
    print("=== Running best QRDQN model ===")
    run(ENV_ID, model_type="QRDQN", model_path=model_path)


