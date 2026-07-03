# Dump per-step policy observations + actions from the Isaac env (env 0)
# for numerical comparison against the MuJoCo sim2sim reimplementation.

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--out", type=str, default="obs_dump.npz")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import go2_backflip.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    import cli_args as _cli
    agent_cfg = _cli.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.reward_override_file = ""
    # disable obs noise so the dump is deterministic and comparable
    env_cfg.obs_noise_ang_vel = 0.0
    env_cfg.obs_noise_gravity = 0.0
    env_cfg.obs_noise_joint_pos = 0.0
    env_cfg.obs_noise_joint_vel = 0.0
    env_cfg.action_delay_substeps_max = 0

    import os
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else get_checkpoint_path(log_root_path)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs_log, act_log = [], []
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            obs_log.append(obs["policy"][0].cpu().numpy().copy())
            actions = policy(obs)
            act_log.append(actions[0].cpu().numpy().copy())
            obs, _, _, _ = env.step(actions)
    np.savez(args_cli.out, obs=np.array(obs_log), act=np.array(act_log))
    print(f"[dump] saved {len(obs_log)} steps -> {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
