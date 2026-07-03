# Go2 backflip direct RL environment.
#
# MDP formulation: observations include a phase variable (episode progress) so that
# the time-windowed rewards remain a function of the state (Markov property).

from __future__ import annotations

import json
import os

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .go2_backflip_env_cfg import Go2BackflipEnvCfg

# joint mirror pairs: (left, right), hip abduction flips sign
_LEFT_JOINTS = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
_RIGHT_JOINTS = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
_MIRROR_SIGN = [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0]


class Go2BackflipEnv(DirectRLEnv):
    cfg: Go2BackflipEnvCfg

    def __init__(self, cfg: Go2BackflipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        num_actions = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, num_actions, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, num_actions, device=self.device)

        # integrated pitch angle (rad); backward flip accumulates negative values
        self._pitch_int = torch.zeros(self.num_envs, device=self.device)
        # per-episode physical metrics (ground truth, independent of reward scales)
        self._max_height = torch.zeros(self.num_envs, device=self.device)
        self._max_up_vel = torch.zeros(self.num_envs, device=self.device)
        self._max_impact = torch.zeros(self.num_envs, device=self.device)

        # sim2real: per-env action delay (physics substeps), motor zero offset, delay buffers
        self._action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._motor_offset = torch.zeros(self.num_envs, num_actions, device=self.device)
        self._processed_actions = self._robot.data.default_joint_pos.clone()
        self._prev_targets = self._robot.data.default_joint_pos.clone()
        self._substep = 0

        # body/joint indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_contact_ids, _ = self._contact_sensor.find_bodies(
            ["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
        )
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(["base", ".*_thigh", ".*_hip"])
        self._feet_body_ids, _ = self._robot.find_bodies(
            ["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
        )
        self._left_joint_ids, _ = self._robot.find_joints(_LEFT_JOINTS, preserve_order=True)
        self._right_joint_ids, _ = self._robot.find_joints(_RIGHT_JOINTS, preserve_order=True)
        self._mirror_sign = torch.tensor(_MIRROR_SIGN, device=self.device)

        # live reward tuning: watch a JSON file and hot-reload weights during training
        self._init_reward_overrides()

        # phase window masks are recomputed every step from episode time
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "up_vel",
                "flip_ang_vel",
                "orientation_control",
                "height_control",
                "flat_orientation",
                "roll",
                "yaw_ang_vel",
                "symmetry",
                "feet_height_pre",
                "feet_distance",
                "action_rate",
                "dof_pos_limits",
                "undesired_contacts",
                "torques",
                "dof_acc",
                "landing",
                "drift",
                "impact",
            ]
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._prev_targets = self._processed_actions
        self._processed_actions = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos + self._motor_offset
        )
        self._substep = 0

    def _apply_action(self):
        # sim2real action latency: delayed envs keep tracking the previous target for
        # their first `_action_delay` physics substeps of each control step
        delayed = (self._substep < self._action_delay).unsqueeze(-1)
        self._robot.set_joint_position_target(torch.where(delayed, self._prev_targets, self._processed_actions))
        self._substep += 1

    # -- live reward tuning --------------------------------------------------

    def _init_reward_overrides(self):
        """Create the override file with this stage's values and start watching it.

        Set cfg.reward_override_file to "" (e.g. in capture/play scripts) to disable —
        otherwise a second process would clobber the file watched by a live training run.
        """
        if not self.cfg.reward_override_file:
            self._override_path = None
            return
        self._tunable_keys = sorted(
            [k for k in dir(self.cfg) if k.endswith("_reward_scale")]
            + [
                "target_pitch",
                "flip_ang_vel_max",
                "up_vel_clamp",
                "target_height",
                "feet_distance_target",
                "obs_noise_ang_vel",
                "obs_noise_gravity",
                "obs_noise_joint_pos",
                "obs_noise_joint_vel",
                "action_delay_substeps_max",
                "prep_end_s",
                "takeoff_start_s",
                "takeoff_end_s",
                "flip_start_s",
                "flip_end_s",
                "land_start_s",
            ]
        )
        self._override_path = os.path.abspath(self.cfg.reward_override_file)
        self._override_mtime = 0.0
        try:
            # always rewrite at run start so stale values from a previous stage never leak in
            with open(self._override_path, "w") as f:
                json.dump({k: getattr(self.cfg, k) for k in self._tunable_keys}, f, indent=2)
            self._override_mtime = os.path.getmtime(self._override_path)
            print(f"[go2_backflip] 실시간 보상 튜닝 파일: {self._override_path} (학습 중 편집하면 즉시 반영)")
        except OSError as err:
            print(f"[go2_backflip] 보상 튜닝 파일 생성 실패({err}) — 실시간 튜닝 비활성화")

    def _reload_reward_overrides(self):
        """Apply the override file if it changed. Called once per episode batch reset."""
        if self._override_path is None:
            return
        try:
            mtime = os.path.getmtime(self._override_path)
        except OSError:
            return
        if mtime <= self._override_mtime:
            return
        self._override_mtime = mtime
        try:
            with open(self._override_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as err:
            print(f"[go2_backflip] reward_overrides.json 읽기 실패({err}) — 이번 변경 무시")
            return
        changed = []
        for key, value in data.items():
            if key in self._tunable_keys and isinstance(value, (int, float)):
                if getattr(self.cfg, key) != float(value):
                    setattr(self.cfg, key, float(value))
                    changed.append(f"{key}={value}")
        if changed:
            print(f"[go2_backflip] 보상/커리큘럼 실시간 변경 적용: {', '.join(changed)}")

    # -- helpers ------------------------------------------------------------

    def _episode_time_s(self) -> torch.Tensor:
        return self.episode_length_buf.float() * self.step_dt

    def _target_pitch_ref(self, t: torch.Tensor) -> torch.Tensor:
        """Pitch reference ramp: 0 -> -target_pitch over the flip window, then held."""
        ramp = (t - self.cfg.flip_start_s) / (self.cfg.flip_end_s - self.cfg.flip_start_s)
        ramp = torch.clamp(ramp, 0.0, 1.0)
        return -self.cfg.target_pitch * ramp

    # -- MDP ----------------------------------------------------------------

    @staticmethod
    def _noisy(x: torch.Tensor, scale: float) -> torch.Tensor:
        return x + (2.0 * torch.rand_like(x) - 1.0) * scale

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        phase = (self._episode_time_s() / self.max_episode_length_s).unsqueeze(-1)
        # policy obs: deployable (no base lin vel) + sensor noise
        obs = torch.cat(
            [
                self._noisy(self._robot.data.root_ang_vel_b, self.cfg.obs_noise_ang_vel),
                self._noisy(self._robot.data.projected_gravity_b, self.cfg.obs_noise_gravity),
                self._noisy(
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos, self.cfg.obs_noise_joint_pos
                ),
                self._noisy(self._robot.data.joint_vel, self.cfg.obs_noise_joint_vel),
                self._actions,
                phase,
            ],
            dim=-1,
        )
        # critic obs: privileged, noise-free (asymmetric actor-critic)
        feet_forces = torch.max(
            torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_ids], dim=-1), dim=1
        )[0]
        feet_contact = (feet_forces > 1.0).float()
        base_height = (self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]).unsqueeze(-1)
        state = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._actions,
                phase,
                feet_contact,
                base_height,
            ],
            dim=-1,
        )
        return {"policy": obs, "critic": state}

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        t = self._episode_time_s()

        # phase window masks
        in_prep = t < cfg.prep_end_s
        in_takeoff = (t >= cfg.takeoff_start_s) & (t < cfg.takeoff_end_s)
        in_flip = (t >= cfg.flip_start_s) & (t < cfg.flip_end_s)
        in_land = t >= cfg.land_start_s

        # integrate pitch rate (body-frame y angular velocity)
        ang_vel_y = self._robot.data.root_ang_vel_b[:, 1]
        self._pitch_int += ang_vel_y * self.step_dt

        # r_up: upward velocity during takeoff (world frame)
        up_vel = torch.clamp(self._robot.data.root_lin_vel_w[:, 2], 0.0, cfg.up_vel_clamp) * in_takeoff

        # r_flip: backward pitch rate (negative ang_vel_y) during flip window
        flip_ang_vel = torch.clamp(-ang_vel_y, 0.0, cfg.flip_ang_vel_max) * in_flip

        # r_ori: track pitch reference ramp
        pitch_err = torch.square(self._pitch_int - self._target_pitch_ref(t))

        # r_height: base height target outside the flip window
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        height_err = torch.abs(base_height - cfg.target_height) * (~in_flip)

        # physical metrics for promotion/verification (not part of the reward)
        self._max_height = torch.maximum(self._max_height, base_height)
        self._max_up_vel = torch.maximum(self._max_up_vel, self._robot.data.root_lin_vel_w[:, 2])

        # r_flat: keep body level (Stage 1 only)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # r_roll / r_yaw
        roll = torch.abs(self._robot.data.projected_gravity_b[:, 1])
        yaw_ang_vel = torch.square(self._robot.data.root_ang_vel_b[:, 2])

        # r_sym: left-right action symmetry (hip abduction mirrored)
        act_left = self._actions[:, self._left_joint_ids] * self._mirror_sign
        act_right = self._actions[:, self._right_joint_ids]
        symmetry = torch.sum(torch.square(act_left - act_right), dim=1)

        # r_feet_pre: feet must stay on the ground before takeoff
        feet_z = self._robot.data.body_pos_w[:, self._feet_body_ids, 2] - self._terrain.env_origins[:, 2].unsqueeze(1)
        feet_height_pre = torch.sum(torch.clamp(feet_z - 0.03, min=0.0), dim=1) * in_prep

        # r_feet_dist: front/rear stance width
        d_front = torch.norm(
            self._robot.data.body_pos_w[:, self._feet_body_ids[0]] - self._robot.data.body_pos_w[:, self._feet_body_ids[1]],
            dim=-1,
        )
        d_rear = torch.norm(
            self._robot.data.body_pos_w[:, self._feet_body_ids[2]] - self._robot.data.body_pos_w[:, self._feet_body_ids[3]],
            dim=-1,
        )
        feet_distance = torch.abs(d_front - cfg.feet_distance_target) + torch.abs(d_rear - cfg.feet_distance_target)

        # r_rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # r_limit: soft joint position limits
        soft_limits = self._robot.data.soft_joint_pos_limits
        out_of_limits = -(self._robot.data.joint_pos - soft_limits[..., 0]).clip(max=0.0)
        out_of_limits += (self._robot.data.joint_pos - soft_limits[..., 1]).clip(min=0.0)
        dof_pos_limits = torch.sum(out_of_limits, dim=1)

        # r_contact: undesired body contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        undesired_contacts = torch.sum(is_contact, dim=1).float()

        # r_torque / r_acc (Stage 4)
        torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        dof_acc = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        # r_land: all four feet in contact + level body + pitch target reached (landing window)
        feet_forces = torch.max(torch.norm(net_contact_forces[:, :, self._feet_contact_ids], dim=-1), dim=1)[0]
        all_feet_contact = torch.all(feet_forces > 1.0, dim=1)

        # r_drift (Stage 5): land where you took off — xy distance from env origin, land window only
        xy_drift = torch.norm(
            self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2], dim=-1
        )
        drift = torch.square(xy_drift) * in_land

        # r_impact (Stage 5): penalize per-foot contact force above threshold (soft landing)
        impact_excess = torch.sum(torch.clamp(feet_forces - cfg.impact_force_threshold, min=0.0), dim=1)
        self._max_impact = torch.maximum(self._max_impact, torch.max(feet_forces, dim=1)[0])
        is_level = flat_orientation < 0.04
        pitch_done = torch.abs(self._pitch_int + cfg.target_pitch) < 0.3
        landing = (all_feet_contact & is_level & pitch_done & in_land).float()

        rewards = {
            "up_vel": up_vel * cfg.up_vel_reward_scale * self.step_dt,
            "flip_ang_vel": flip_ang_vel * cfg.flip_ang_vel_reward_scale * self.step_dt,
            "orientation_control": pitch_err * cfg.orientation_control_reward_scale * self.step_dt,
            "height_control": height_err * cfg.height_control_reward_scale * self.step_dt,
            "flat_orientation": flat_orientation * cfg.flat_orientation_reward_scale * self.step_dt,
            "roll": roll * cfg.roll_reward_scale * self.step_dt,
            "yaw_ang_vel": yaw_ang_vel * cfg.yaw_ang_vel_reward_scale * self.step_dt,
            "symmetry": symmetry * cfg.symmetry_reward_scale * self.step_dt,
            "feet_height_pre": feet_height_pre * cfg.feet_height_pre_reward_scale * self.step_dt,
            "feet_distance": feet_distance * cfg.feet_distance_reward_scale * self.step_dt,
            "action_rate": action_rate * cfg.action_rate_reward_scale * self.step_dt,
            "dof_pos_limits": dof_pos_limits * cfg.dof_pos_limits_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * cfg.undesired_contact_reward_scale * self.step_dt,
            "torques": torques * cfg.torque_reward_scale * self.step_dt,
            "dof_acc": dof_acc * cfg.dof_acc_reward_scale * self.step_dt,
            "landing": landing * cfg.landing_reward_scale * self.step_dt,
            "drift": drift * cfg.drift_reward_scale * self.step_dt,
            "impact": impact_excess * cfg.impact_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # CMDP hard constraint (Constraints-as-Terminations): violating the impact
        # limit forfeits all future reward — not tradable against other rewards.
        if self.cfg.constraint_impact_force_max > 0.0:
            feet_forces = torch.max(
                torch.norm(
                    self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_ids], dim=-1
                ),
                dim=1,
            )[0]
            died = torch.any(feet_forces > self.cfg.constraint_impact_force_max, dim=1)
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # hot-reload reward weights if the override file was edited (live tuning)
        self._reload_reward_overrides()
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        # NOTE: unlike locomotion tasks, episode_length_buf must NOT be randomized here —
        # the phase-timed rewards require every episode to start at t=0.
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        final_pitch_int = torch.mean(self._pitch_int[env_ids]).item()
        max_height = torch.mean(self._max_height[env_ids]).item()
        max_up_vel = torch.mean(self._max_up_vel[env_ids]).item()
        max_impact = torch.mean(self._max_impact[env_ids]).item()
        final_drift = torch.mean(
            torch.norm(self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=-1)
        ).item()
        self._pitch_int[env_ids] = 0.0
        self._max_height[env_ids] = 0.0
        self._max_up_vel[env_ids] = 0.0
        self._max_impact[env_ids] = 0.0
        # sim2real: resample per-env action delay and motor zero offset
        self._action_delay[env_ids] = torch.randint(
            0, int(self.cfg.action_delay_substeps_max) + 1, (len(env_ids),), device=self.device
        )
        lo, hi = self.cfg.motor_offset_range
        self._motor_offset[env_ids] = torch.rand_like(self._motor_offset[env_ids]) * (hi - lo) + lo
        self._processed_actions[env_ids] = self._robot.data.default_joint_pos[env_ids]
        self._prev_targets[env_ids] = self._robot.data.default_joint_pos[env_ids]
        # reset robot to default standing state (+ randomization to avoid overfitting
        # the policy to one engine's exact spawn transient — sim2sim lesson)
        def _unoise(shape, scale):
            return (torch.rand(shape, device=self.device) * 2.0 - 1.0) * scale

        joint_pos = self._robot.data.default_joint_pos[env_ids] + _unoise(
            self._robot.data.default_joint_pos[env_ids].shape, self.cfg.reset_joint_pos_noise
        )
        joint_vel = self._robot.data.default_joint_vel[env_ids] + _unoise(
            self._robot.data.default_joint_vel[env_ids].shape, self.cfg.reset_joint_vel_noise
        )
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] += _unoise((len(env_ids),), self.cfg.reset_height_noise)
        default_root_state[:, 10:13] += _unoise((len(env_ids), 3), self.cfg.reset_ang_vel_noise)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        # curriculum diagnostics: how much rotation was achieved + live-tunable values
        extras["Episode_Metric/final_pitch_int"] = final_pitch_int
        extras["Episode_Metric/max_base_height"] = max_height
        extras["Episode_Metric/max_up_vel"] = max_up_vel
        extras["Episode_Metric/max_impact_force"] = max_impact
        extras["Episode_Metric/final_xy_drift"] = final_drift
        extras["Curriculum/target_pitch"] = self.cfg.target_pitch
        extras["Curriculum/flip_ang_vel_max"] = self.cfg.flip_ang_vel_max
        extras["Curriculum/flip_ang_vel_reward_scale"] = self.cfg.flip_ang_vel_reward_scale
        extras["Curriculum/undesired_contact_reward_scale"] = self.cfg.undesired_contact_reward_scale
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
