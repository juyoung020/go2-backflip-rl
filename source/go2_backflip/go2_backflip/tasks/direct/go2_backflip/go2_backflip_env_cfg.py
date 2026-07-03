# Go2 backflip environment configurations (Stage 1-4 curriculum).
#
# Reward design based on Genesis-backflip (https://github.com/ziyanx02/Genesis-backflip)
# adapted to IsaacLab direct workflow. See plan: curriculum stages switch only the
# reward weights and curriculum variables (flip_ang_vel_max, target_pitch).

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

import math


@configclass
class BaseEventCfg:
    """Fixed physics parameters (no randomization)."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (0.8, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class RandomizedEventCfg(BaseEventCfg):
    """Domain randomization for Stage 4 (Genesis-backflip ranges)."""

    def __post_init__(self):
        self.physics_material.params["static_friction_range"] = (0.5, 1.25)
        self.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )


@configclass
class Go2BackflipEnvCfg(DirectRLEnvCfg):
    """Base config. Stage classes below override reward weights / curriculum variables."""

    # env
    episode_length_s = 2.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    # policy obs (deployable, noisy): ang_vel(3) + gravity(3) + joint_pos(12) + joint_vel(12)
    #   + actions(12) + phase(1) — NO base lin vel (not measurable on the real robot)
    observation_space = 43
    # critic obs (privileged, clean): lin_vel(3) + ang_vel(3) + gravity(3) + joint_pos(12)
    #   + joint_vel(12) + actions(12) + phase(1) + feet_contact(4) + base_height(1)
    state_space = 51

    # -- sim2real: observation noise (uniform, added to policy obs only) --
    obs_noise_ang_vel = 0.2      # rad/s
    obs_noise_gravity = 0.05
    obs_noise_joint_pos = 0.01   # rad
    obs_noise_joint_vel = 1.5    # rad/s

    # -- sim2real: action latency & motor offset --
    # per-env action delay in physics substeps (1 substep = 5ms), resampled at reset
    action_delay_substeps_max = 2  # 0~10ms
    # per-env joint encoder/actuation zero offset (rad), resampled at reset
    motor_offset_range = (-0.02, 0.02)

    # simulation: 200Hz physics, 50Hz control
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # events
    events: BaseEventCfg = BaseEventCfg()

    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # -- live reward tuning --
    # JSON file watched during training; edit it while a run is active to change
    # reward weights / curriculum variables in real time (no restart needed).
    # The file is rewritten with the current stage's values at every run start.
    reward_override_file = "reward_overrides.json"

    # -- phase timing (seconds) --
    prep_end_s = 0.5       # crouch/preparation window [0, prep_end)
    takeoff_start_s = 0.5  # upward-velocity reward window
    takeoff_end_s = 0.75
    flip_start_s = 0.5     # rotation reward window + pitch ramp
    flip_end_s = 1.0
    land_start_s = 1.0     # landing posture bonus window

    # -- curriculum variables --
    target_pitch = 0.0          # total backward rotation (rad); pi=180deg, 2*pi=full backflip
    flip_ang_vel_max = 0.0      # clamp for backward pitch rate reward (rad/s)

    # -- reward scales (per-second; multiplied by step_dt in env) --
    up_vel_reward_scale = 20.0            # r_up
    up_vel_clamp = 3.0                    # m/s
    flip_ang_vel_reward_scale = 0.0       # r_flip (enabled from Stage 2)
    orientation_control_reward_scale = 0.0  # r_ori (enabled from Stage 2)
    height_control_reward_scale = -10.0   # r_height
    target_height = 0.3                   # m
    flat_orientation_reward_scale = -5.0  # r_flat (Stage 1 only)
    roll_reward_scale = -10.0             # r_roll
    yaw_ang_vel_reward_scale = -1.0       # r_yaw
    symmetry_reward_scale = -0.1          # r_sym
    feet_height_pre_reward_scale = -30.0  # r_feet_pre
    feet_distance_reward_scale = -1.0     # r_feet_dist
    feet_distance_target = 0.3            # m
    action_rate_reward_scale = -0.001     # r_rate
    dof_pos_limits_reward_scale = -10.0   # r_limit
    undesired_contact_reward_scale = -1.0  # r_contact
    torque_reward_scale = 0.0             # r_torque (Stage 4)
    dof_acc_reward_scale = 0.0            # r_acc (Stage 4)
    landing_reward_scale = 0.0            # r_land (Stage 4)
    drift_reward_scale = 0.0              # r_drift (Stage 5): land in place
    impact_reward_scale = 0.0             # r_impact (Stage 5): soft landing
    impact_force_threshold = 300.0        # N (~2x body weight); force above this is penalized
    # -- sim2sim: reset-state randomization (fixed resets overfit the policy to
    #    the physics engine's exact initial transient — found via MuJoCo sim2sim) --
    reset_height_noise = 0.0      # +- m on initial base z (nominal 0.4)
    reset_joint_pos_noise = 0.0   # +- rad on initial joint positions
    reset_joint_vel_noise = 0.0   # +- rad/s on initial joint velocities
    reset_ang_vel_noise = 0.0     # +- rad/s on initial base angular velocity

    # CMDP hard constraint (Constraints-as-Terminations): episode ends immediately
    # if any foot contact force exceeds this. 0 = disabled. Go2 publishes no impact
    # rating; 700 N (~4.8x body weight) is set from quadruped landing literature.
    constraint_impact_force_max = 0.0


@configclass
class Go2BackflipStage1Cfg(Go2BackflipEnvCfg):
    """Stage 1 — vertical jump in place, no rotation."""

    pass


@configclass
class Go2BackflipStage2Cfg(Go2BackflipEnvCfg):
    """Stage 2 — partial backward rotation (180 deg), back landing allowed."""

    target_pitch = math.pi
    flip_ang_vel_max = 3.6
    flip_ang_vel_reward_scale = 5.0
    orientation_control_reward_scale = -1.0
    flat_orientation_reward_scale = 0.0
    undesired_contact_reward_scale = -0.1


@configclass
class Go2BackflipStage3Cfg(Go2BackflipEnvCfg):
    """Stage 3 — full 360 deg backflip, land on feet."""

    target_pitch = 2.0 * math.pi
    flip_ang_vel_max = 7.2
    flip_ang_vel_reward_scale = 5.0
    orientation_control_reward_scale = -1.0
    flat_orientation_reward_scale = 0.0
    undesired_contact_reward_scale = -1.0


@configclass
class Go2BackflipStage4Cfg(Go2BackflipStage3Cfg):
    """Stage 4 — landing refinement + robustness (domain randomization)."""

    events: RandomizedEventCfg = RandomizedEventCfg()
    torque_reward_scale = -0.0002
    dof_acc_reward_scale = -2.5e-7
    landing_reward_scale = 10.0


@configclass
class Go2BackflipStage5Cfg(Go2BackflipStage4Cfg):
    """Stage 5 — precision & soft landing: land in place, absorb impact."""

    drift_reward_scale = -5.0     # xy distance from start, land window only
    impact_reward_scale = -0.005  # per-N above threshold (477N peak -> ~0.9/step penalty)
    torque_reward_scale = -0.0005  # stronger torque suppression than Stage 4
    dof_acc_reward_scale = -1.0e-6
    # reset randomization: break dependence on the exact spawn transient (sim2sim)
    reset_height_noise = 0.05
    reset_joint_pos_noise = 0.1
    reset_joint_vel_noise = 0.5
    reset_ang_vel_noise = 0.5


@configclass
class Go2BackflipStage6Cfg(Go2BackflipStage5Cfg):
    """Stage 6 — CMDP: impact hard constraint via termination (run if Stage 5
    soft penalties fail to bring peak impact under target)."""

    constraint_impact_force_max = 700.0
