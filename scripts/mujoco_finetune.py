# Fine-tune the Go2 backflip policy in MuJoCo (professor pipeline step 2:
# IsaacGym-train -> MuJoCo-retrain -> real controller).
#
# Reimplements the Go2BackflipEnv MDP (obs 43/critic 51, 16 reward terms,
# Stage-5 weights, PD + DC-motor curve, reset randomization) on MuJoCo and
# fine-tunes the Isaac checkpoint with rsl_rl PPO on CPU.
#
# Run:
#   .venv-mj/bin/python scripts/mujoco_finetune.py \
#       --checkpoint logs/rsl_rl/go2_backflip/2026-07-03_18-04-57/model_5298.pt \
#       --num_envs 128 --iterations 400

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner

ISAAC_JOINTS = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
DEFAULT_Q = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])

# Stage 5 reward weights (kept identical to Go2BackflipStage5Cfg)
W = dict(
    up_vel=20.0, flip=5.0, ori=-1.0, height=-10.0, roll=-10.0, yaw=-1.0,
    sym=-0.1, feet_pre=-30.0, feet_dist=-1.0, rate=-0.001, limit=-10.0,
    contact=-1.0, torque=-0.0005, acc=-1.0e-6, land=10.0, drift=-5.0, impact=-0.005,
)
TARGET_PITCH = 2.0 * math.pi
FLIP_W_MAX, UP_CLAMP = 7.2, 3.0
PREP_END, TO_START, TO_END, FLIP_START, FLIP_END, LAND_START = 0.5, 0.5, 0.75, 0.5, 1.0, 1.0
KP, KD, SAT, EFF, VMAX = 25.0, 0.5, 23.5, 23.5, 30.0
CTRL_DT, EPISODE_S = 0.02, 2.0
IMPACT_THR = 300.0


class MujocoBackflipVecEnv:
    """rsl_rl VecEnv on N independent MuJoCo instances (threaded stepping)."""

    def __init__(self, xml, num_envs, seed=0):
        self.num_envs = num_envs
        self.num_actions = 12
        self.device = torch.device("cpu")
        self.max_episode_length = int(EPISODE_S / CTRL_DT)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.cfg = {}
        self.rng = np.random.default_rng(seed)

        base = mujoco.MjModel.from_xml_path(xml)
        for n in ISAAC_JOINTS:
            dof = base.joint(n).dofadr[0]
            base.dof_damping[dof] = 0.0
            base.dof_frictionloss[dof] = 0.0
        for g in range(base.ngeom):
            if base.geom(g).name in ("FL", "FR", "RL", "RR"):
                base.geom_friction[g, 0] = 1.0
        self.model = base
        self.n_sub = round(CTRL_DT / base.opt.timestep)
        self.data = [mujoco.MjData(base) for _ in range(num_envs)]
        self.qadr = np.array([base.joint(n).qposadr[0] for n in ISAAC_JOINTS])
        self.dadr = np.array([base.joint(n).dofadr[0] for n in ISAAC_JOINTS])
        self.aid = np.array([base.actuator(n.replace("_joint", "")).id for n in ISAAC_JOINTS])
        self.foot_bodies = [base.body(f"{l}_calf").id for l in ["FL", "FR", "RL", "RR"]]  # feet geoms sit on calf
        self.foot_geoms = [mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("FL", "FR", "RL", "RR")]

        self.actions = np.zeros((num_envs, 12), dtype=np.float32)
        self.prev_actions = np.zeros_like(self.actions)
        self.pitch_int = np.zeros(num_envs)
        self.pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.ep_rew = np.zeros(num_envs)
        for i in range(num_envs):
            self._reset_one(i)

    # -- helpers --
    def _grav_b(self, d):
        q = d.qpos[3:7]
        qi = np.zeros(4); mujoco.mju_negQuat(qi, q)
        out = np.zeros(3); mujoco.mju_rotVecQuat(out, np.array([0.0, 0.0, -1.0]), qi)
        return out

    def _foot_forces(self, d):
        # per-foot contact force magnitude via contact iteration
        f = np.zeros(4)
        buf = np.zeros(6)
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            for k, fg in enumerate(self.foot_geoms):
                if g1 == fg or g2 == fg:
                    mujoco.mj_contactForce(self.model, d, c, buf)
                    f[k] += np.linalg.norm(buf[:3])
        return f

    def _reset_one(self, i):
        d = self.data[i]
        mujoco.mj_resetData(self.model, d)
        d.qpos[:3] = [0.0, 0.0, 0.4 + self.rng.uniform(-0.05, 0.05)]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[self.qadr] = DEFAULT_Q + self.rng.uniform(-0.1, 0.1, 12)
        d.qvel[:] = 0.0
        d.qvel[self.dadr] = self.rng.uniform(-0.5, 0.5, 12)
        d.qvel[3:6] = self.rng.uniform(-0.5, 0.5, 3)
        mujoco.mj_forward(self.model, d)
        self.actions[i] = 0.0
        self.prev_actions[i] = 0.0
        self.pitch_int[i] = 0.0
        self.episode_length_buf[i] = 0
        self.ep_rew[i] = 0.0

    def _obs_one(self, i):
        d = self.data[i]
        t = self.episode_length_buf[i].item() * CTRL_DT
        grav = self._grav_b(d)
        jp = d.qpos[self.qadr] - DEFAULT_Q
        jv = d.qvel[self.dadr]
        policy = np.concatenate([d.qvel[3:6], grav, jp, jv, self.actions[i], [t / EPISODE_S]])
        feet = (self._foot_forces(d) > 1.0).astype(np.float64)
        critic = np.concatenate([d.qvel[0:3], d.qvel[3:6], grav, jp, jv, self.actions[i], [t / EPISODE_S], feet, [d.qpos[2]]])
        return policy, critic

    def get_observations(self):
        po, cr = zip(*[self._obs_one(i) for i in range(self.num_envs)])
        return TensorDict(
            {"policy": torch.tensor(np.array(po), dtype=torch.float32),
             "critic": torch.tensor(np.array(cr), dtype=torch.float32)},
            batch_size=[self.num_envs],
        )

    def _step_one(self, i):
        d = self.data[i]
        q_target = DEFAULT_Q + 0.5 * self.actions[i]
        for _ in range(self.n_sub):
            jv = d.qvel[self.dadr]
            tau = KP * (q_target - d.qpos[self.qadr]) - KD * jv
            tmax = np.clip(SAT * (1.0 - jv / VMAX), 0.0, EFF)
            tmin = np.clip(-SAT * (1.0 + jv / VMAX), -EFF, 0.0)
            d.ctrl[self.aid] = np.clip(tau, tmin, tmax)
            mujoco.mj_step(self.model, d)
        return self._reward_one(i)

    def _reward_one(self, i):
        d = self.data[i]
        t = self.episode_length_buf[i].item() * CTRL_DT
        in_prep, in_to = t < PREP_END, TO_START <= t < TO_END
        in_flip, in_land = FLIP_START <= t < FLIP_END, t >= LAND_START
        wy = d.qvel[4]
        self.pitch_int[i] += wy * CTRL_DT
        grav = self._grav_b(d)
        h = d.qpos[2]
        jp = d.qpos[self.qadr]
        feet_f = self._foot_forces(d)
        # foot z: use geom xpos
        foot_z = np.array([d.geom_xpos[g][2] for g in self.foot_geoms])
        a, pa = self.actions[i], self.prev_actions[i]
        left, right = [0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]  # FL,RL | FR,RR per type group
        mirror = np.array([-1, -1, 1, 1, 1, 1])                 # hips flip sign
        ramp = np.clip((t - FLIP_START) / (FLIP_END - FLIP_START), 0.0, 1.0)
        theta_ref = -TARGET_PITCH * ramp
        tau_now = d.ctrl[self.aid]
        r = 0.0
        r += W["up_vel"] * (np.clip(d.qvel[2], 0, UP_CLAMP) if in_to else 0.0)
        r += W["flip"] * (np.clip(-wy, 0, FLIP_W_MAX) if in_flip else 0.0)
        r += W["ori"] * (self.pitch_int[i] - theta_ref) ** 2
        r += W["height"] * (abs(h - 0.3) if not in_flip else 0.0)
        r += W["roll"] * abs(grav[1])
        r += W["yaw"] * d.qvel[5] ** 2
        r += W["sym"] * float(np.sum((a[left] * mirror - a[right]) ** 2))
        r += W["feet_pre"] * (float(np.sum(np.clip(foot_z - 0.03, 0, None))) if in_prep else 0.0)
        d_front = np.linalg.norm(d.geom_xpos[self.foot_geoms[0]] - d.geom_xpos[self.foot_geoms[1]])
        d_rear = np.linalg.norm(d.geom_xpos[self.foot_geoms[2]] - d.geom_xpos[self.foot_geoms[3]])
        r += W["feet_dist"] * (abs(d_front - 0.3) + abs(d_rear - 0.3))
        r += W["rate"] * float(np.sum((a - pa) ** 2))
        lo, hi = self.model.jnt_range[[self.model.joint(n).id for n in ISAAC_JOINTS], 0], \
                 self.model.jnt_range[[self.model.joint(n).id for n in ISAAC_JOINTS], 1]
        soft_lo, soft_hi = lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)
        r += W["limit"] * float(np.sum(np.clip(soft_lo - jp, 0, None) + np.clip(jp - soft_hi, 0, None)))
        # undesired contact: trunk geom contact
        trunk_hit = 0.0
        for c in range(self.data[i].ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            b1, b2 = self.model.geom_bodyid[g1], self.model.geom_bodyid[g2]
            names = [self.model.body(b1).name, self.model.body(b2).name]
            if any(("base" in n or "trunk" in n or "hip" in n or "thigh" in n) for n in names):
                trunk_hit = 1.0
        r += W["contact"] * trunk_hit
        r += W["torque"] * float(np.sum(tau_now ** 2))
        all_feet = np.all(feet_f > 1.0)
        level = (grav[0] ** 2 + grav[1] ** 2) < 0.04
        pitch_done = abs(self.pitch_int[i] + TARGET_PITCH) < 0.3
        r += W["land"] * (1.0 if (all_feet and level and pitch_done and in_land) else 0.0)
        r += W["drift"] * (float(np.sum(d.qpos[:2] ** 2)) if in_land else 0.0)
        r += W["impact"] * float(np.sum(np.clip(feet_f - IMPACT_THR, 0, None)))
        return r * CTRL_DT

    def step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.copy()
        self.actions = actions.cpu().numpy().astype(np.float32)
        rewards = np.array(list(self.pool.map(self._step_one, range(self.num_envs))))
        self.episode_length_buf += 1
        self.ep_rew += rewards
        dones = self.episode_length_buf >= self.max_episode_length
        time_outs = dones.clone()
        extras = {"log": {}, "time_outs": time_outs}
        if dones.any():
            idx = torch.nonzero(dones).flatten().tolist()
            extras["log"]["Episode_Metric/final_pitch_int"] = float(np.mean(self.pitch_int[idx]))
            extras["log"]["Episode_Reward/total"] = float(np.mean(self.ep_rew[idx]))
            for i in idx:
                self._reset_one(i)
        obs = self.get_observations()
        return obs, torch.tensor(rewards, dtype=torch.float32), dones.to(torch.long), extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--xml", default="mujoco_menagerie/unitree_go2/scene.xml")
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--logdir", default="logs/mujoco_finetune")
    ap.add_argument("--up-clamp", type=float, default=None)
    ap.add_argument("--impact-w", type=float, default=None)
    ap.add_argument("--impact-thr", type=float, default=None)
    ap.add_argument("--flip-w", type=float, default=None)
    args = ap.parse_args()

    global UP_CLAMP, IMPACT_THR
    if args.up_clamp is not None: UP_CLAMP = args.up_clamp
    if args.impact_thr is not None: IMPACT_THR = args.impact_thr
    if args.impact_w is not None: W["impact"] = args.impact_w
    if args.flip_w is not None: W["flip"] = args.flip_w
    print(f"[finetune] UP_CLAMP={UP_CLAMP} impact_w={W['impact']} thr={IMPACT_THR} flip_w={W['flip']}")

    env = MujocoBackflipVecEnv(args.xml, args.num_envs)
    train_cfg = {
        "num_steps_per_env": 24, "save_interval": 50, "empirical_normalization": None,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {"class_name": "ActorCritic", "init_noise_std": 0.3,
                   "actor_obs_normalization": False, "critic_obs_normalization": False,
                   "actor_hidden_dims": [512, 256, 128], "critic_hidden_dims": [512, 256, 128],
                   "activation": "elu"},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True,
                      "clip_param": 0.2, "entropy_coef": 0.005, "num_learning_epochs": 5,
                      "num_mini_batches": 4, "learning_rate": 3.0e-4, "schedule": "adaptive",
                      "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
    }
    os.makedirs(args.logdir, exist_ok=True)
    runner = OnPolicyRunner(env, train_cfg, log_dir=args.logdir, device="cpu")
    runner.load(args.checkpoint, load_optimizer=False, map_location="cpu")  # fresh optimizer for new dynamics
    print(f"[finetune] loaded {args.checkpoint}, envs={args.num_envs}, iters={args.iterations}")
    t0 = time.time()
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    print(f"[finetune] done in {(time.time()-t0)/60:.1f} min -> {args.logdir}")


if __name__ == "__main__":
    main()
