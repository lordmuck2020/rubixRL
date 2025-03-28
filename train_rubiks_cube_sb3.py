import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime

# Import the Rubik's Cube environment
from rubixRL.envs import RubiksCubeEnv


def make_env():
    """Create a wrapped, monitored environment."""
    env = gym.make("RubiksCube-v0", n=3, max_steps=100, reward_type="dense")
    env = Monitor(env, info_keywords=("is_solved",))  # Track if cube is solved
    return env


def main():
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "models"
    run_name = f"sb3_ppo_{timestamp}"
    output_dir = os.path.join(base_dir, run_name)
    tensorboard_dir = os.path.join(base_dir, "tensorboard_logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Create and wrap the environment
    env = DummyVecEnv([make_env])
    env = VecNormalize(
        env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
    )

    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        tensorboard_log=tensorboard_dir,  # Changed to use common tensorboard directory
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
    )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=output_dir, name_prefix="ppo_rubiks_cube"
    )

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
    )

    # Copy statistics from training environment
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/best_model",
        log_path=f"{output_dir}/results",
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    print(f"Tensorboard logs will be saved to: {tensorboard_dir}")
    print("To view training progress, run:")
    print(f"tensorboard --logdir={tensorboard_dir}")
    print("\nStarting training...")

    # Train the model
    total_timesteps = 1_000_000  # Adjust based on your needs
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        tb_log_name=run_name,  # Add unique name for this training run
    )

    # Save the final model and environment normalization
    final_model_path = os.path.join(output_dir, "final_model")
    model.save(final_model_path)
    env.save(os.path.join(output_dir, "vec_normalize.pkl"))

    print(f"\nTraining completed. Models saved to: {output_dir}")
    print(f"Final model saved as: {final_model_path}")

    # Close environments
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
