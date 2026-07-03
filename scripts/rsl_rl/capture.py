# Frame-by-frame motion capture for reward tuning.
#
# Loads a trained checkpoint, rolls out the policy, and records per-control-step
# (50Hz) robot state to CSV: base height, vertical velocity, pitch rate, integrated
# pitch, orientation, per-foot contact, torque peak. Prints a phase-timing summary
# used to verify/adjust the reward time windows.

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Capture per-frame robot motion from a trained policy.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments (env 0 is recorded).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--steps", type=int, default=200, help="Control steps to record (100 = one episode).")
parser.add_argument("--out", type=str, default="capture.csv", help="Output CSV path.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import math
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import go2_backflip.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    # evaluation must never touch the live-tuning file owned by a training run
    env_cfg.reward_override_file = ""

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[capture] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped  # Go2BackflipEnv
    robot = raw._robot
    dt = raw.step_dt

    rows = []
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            i = 0  # recorded env
            contact_forces = raw._contact_sensor.data.net_forces_w_history[i, 0, raw._feet_contact_ids]
            feet_contact = (contact_forces.norm(dim=-1) > 1.0).int().tolist()
            rows.append({
                "step": step,
                "t_s": round((raw.episode_length_buf[i].item()) * dt, 3),
                "base_x": round(robot.data.root_pos_w[i, 0].item() - raw._terrain.env_origins[i, 0].item(), 4),
                "base_y": round(robot.data.root_pos_w[i, 1].item() - raw._terrain.env_origins[i, 1].item(), 4),
                "feet_force_max": round(contact_forces.norm(dim=-1).max().item(), 1),
                "base_h": round(robot.data.root_pos_w[i, 2].item() - raw._terrain.env_origins[i, 2].item(), 4),
                "v_z": round(robot.data.root_lin_vel_w[i, 2].item(), 3),
                "ang_vel_y": round(robot.data.root_ang_vel_b[i, 1].item(), 3),
                "pitch_int": round(raw._pitch_int[i].item(), 3),
                "g_proj_x": round(robot.data.projected_gravity_b[i, 0].item(), 3),
                "g_proj_y": round(robot.data.projected_gravity_b[i, 1].item(), 3),
                "contact_FL": feet_contact[0],
                "contact_FR": feet_contact[1],
                "contact_RL": feet_contact[2],
                "contact_RR": feet_contact[3],
                "torque_max": round(robot.data.applied_torque[i].abs().max().item(), 2),
            })

    with open(args_cli.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[capture] {len(rows)} frames -> {args_cli.out}")

    # ---- phase-timing summary (first episode) ----
    ep = [r for r in rows if r["step"] < 100]
    max_h = max(r["base_h"] for r in ep)
    t_max_h = next(r["t_s"] for r in ep if r["base_h"] == max_h)
    max_vz = max(r["v_z"] for r in ep)
    t_max_vz = next(r["t_s"] for r in ep if r["v_z"] == max_vz)
    min_wy = min(r["ang_vel_y"] for r in ep)
    t_min_wy = next(r["t_s"] for r in ep if r["ang_vel_y"] == min_wy)
    final_pitch = ep[-1]["pitch_int"]
    airborne = [r["t_s"] for r in ep if not any([r["contact_FL"], r["contact_FR"], r["contact_RL"], r["contact_RR"]])]
    torque_peak = max(r["torque_max"] for r in ep)
    print("[capture] ---- episode summary ----")
    print(f"  max base height : {max_h:.3f} m  @ t={t_max_h:.2f}s")
    print(f"  max v_z         : {max_vz:.2f} m/s @ t={t_max_vz:.2f}s")
    print(f"  peak -ang_vel_y : {-min_wy:.2f} rad/s @ t={t_min_wy:.2f}s")
    print(f"  final pitch_int : {final_pitch:.2f} rad (target {-raw.cfg.target_pitch:.2f}, {abs(final_pitch)/(2*math.pi)*360:.0f} deg rotated)")
    if airborne:
        print(f"  airborne window : {min(airborne):.2f}s - {max(airborne):.2f}s ({len(airborne)*dt:.2f}s)")
    else:
        print("  airborne window : none (never left ground)")
    print(f"  torque peak     : {torque_peak:.1f} N*m (limit 23.5)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
