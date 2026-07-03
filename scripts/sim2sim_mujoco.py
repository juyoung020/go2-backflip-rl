# MuJoCo sim2sim validation for the Go2 backflip policy.
#
# Loads the ONNX policy exported from IsaacLab and replays it in MuJoCo
# (menagerie unitree_go2) with the SAME control interface used in training:
# 50 Hz position targets -> 500 Hz PD (kp 25, kd 0.5) -> torque clip 23.5 N*m.
# Observations are rebuilt exactly as in Go2BackflipEnv (43-dim, Isaac joint
# ordering, body-frame ang vel, projected gravity, phase).
#
# Success = policy relies on shared physics, not PhysX quirks.

import argparse
import os

import mujoco
import numpy as np
import onnxruntime as ort

# Isaac PhysX articulation joint order (breadth-first: all hips, thighs, calves)
ISAAC_JOINTS = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
# UNITREE_GO2_CFG default joint positions (rad)
DEFAULT_POS = {
    "FL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
}
KP, KD, TAU_MAX = 25.0, 0.5, 23.5
ACTION_SCALE = 0.5
CTRL_DT = 0.02          # 50 Hz policy
EPISODE_S = 2.0
SETTLE_S = 1.0          # extra time after the episode to judge the landing

parser = argparse.ArgumentParser()
parser.add_argument("--onnx", required=True)
parser.add_argument("--xml", default="mujoco_menagerie/unitree_go2/scene.xml")
parser.add_argument("--video", default="")
parser.add_argument("--joint-order", choices=["type", "leg"], default="type",
                    help="Isaac joint ordering assumption: type-major (default) or leg-major")
args = parser.parse_args()

if args.joint_order == "leg":
    ISAAC_JOINTS = [f"{leg}_{p}_joint" for leg in ["FL", "FR", "RL", "RR"] for p in ["hip", "thigh", "calf"]]

model = mujoco.MjModel.from_xml_path(args.xml)
# --- align model parameters with the Isaac training setup ---
# menagerie bakes joint damping 2.0 + frictionloss 0.2 into the model; training
# had neither (PD Kd 0.5 only). Leaving them in quintuples joint resistance and
# the policy never initiates the flip. Foot friction 0.8 -> 1.0 to match.
for _n in ISAAC_JOINTS:
    _dof = model.joint(_n).dofadr[0]
    model.dof_damping[_dof] = 0.0
    model.dof_frictionloss[_dof] = 0.0
for _g in range(model.ngeom):
    if model.geom(_g).name in ("FL", "FR", "RL", "RR"):
        model.geom_friction[_g, 0] = 1.0
data = mujoco.MjData(model)

# IsaacLab DCMotor speed-torque curve (constant clipping over-powers the motors
# and produced a 534-degree over-rotation before this was replicated)
SAT, VMAX = 23.5, 30.0
def dc_clip(tau, joint_vel):
    tmax = np.clip(SAT * (1.0 - joint_vel / VMAX), 0.0, TAU_MAX)
    tmin = np.clip(-SAT * (1.0 + joint_vel / VMAX), -TAU_MAX, 0.0)
    return np.clip(tau, tmin, tmax)
n_sub = round(CTRL_DT / model.opt.timestep)
print(f"mujoco dt={model.opt.timestep}s, substeps per control step={n_sub}")

qadr = np.array([model.joint(n).qposadr[0] for n in ISAAC_JOINTS])
dadr = np.array([model.joint(n).dofadr[0] for n in ISAAC_JOINTS])
aid = np.array([model.actuator(n.replace("_joint", "")).id for n in ISAAC_JOINTS])
default_q = np.array([DEFAULT_POS[n] for n in ISAAC_JOINTS])

sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
print(f"onnx input: {in_name} {sess.get_inputs()[0].shape}")

# reset to the training initial state: base at 0.4 m, default joints, zero vel
mujoco.mj_resetData(model, data)
data.qpos[:3] = [0.0, 0.0, 0.4]
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qpos[qadr] = default_q
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)

renderer = None
frames = []
if args.video:
    renderer = mujoco.Renderer(model, height=480, width=640)

action = np.zeros(12, dtype=np.float32)
pitch_int = 0.0
peak_impact = 0.0
max_h, max_wy = 0.0, 0.0
log = []

def body_frame(vec_w, quat_wxyz):
    q_inv = np.zeros(4); mujoco.mju_negQuat(q_inv, quat_wxyz)
    out = np.zeros(3); mujoco.mju_rotVecQuat(out, vec_w, q_inv)
    return out

n_steps = int((EPISODE_S + SETTLE_S) / CTRL_DT)
for step in range(n_steps):
    t = step * CTRL_DT
    phase = min(t / EPISODE_S, 1.0)

    quat = data.qpos[3:7].copy()
    ang_vel_b = data.qvel[3:6].copy()          # free-joint angular velocity is body-frame
    grav_b = body_frame(np.array([0.0, 0.0, -1.0]), quat)
    joint_pos = data.qpos[qadr] - default_q
    joint_vel = data.qvel[dadr]

    obs = np.concatenate([
        ang_vel_b, grav_b, joint_pos, joint_vel, action, [phase]
    ]).astype(np.float32)[None, :]

    action = sess.run(None, {in_name: obs})[0][0].astype(np.float32)
    q_target = default_q + ACTION_SCALE * action

    for _ in range(n_sub):
        tau = KP * (q_target - data.qpos[qadr]) - KD * data.qvel[dadr]
        data.ctrl[aid] = dc_clip(tau, data.qvel[dadr])
        mujoco.mj_step(model, data)

    wy = data.qvel[4]
    pitch_int += wy * CTRL_DT
    if t >= 1.0:  # landing window: track peak foot impact
        _buf = np.zeros(6)
        for _c in range(data.ncon):
            _g1, _g2 = data.contact[_c].geom1, data.contact[_c].geom2
            if model.geom(_g1).name in ("FL","FR","RL","RR") or model.geom(_g2).name in ("FL","FR","RL","RR"):
                mujoco.mj_contactForce(model, data, _c, _buf)
                peak_impact = max(peak_impact, float(np.linalg.norm(_buf[:3])))
    max_h = max(max_h, data.qpos[2])
    max_wy = min(max_wy, wy)
    if renderer and step % 1 == 0:
        renderer.update_scene(data, camera=-1)
        frames.append(renderer.render())
    if step % 10 == 0:
        log.append((t, data.qpos[2], wy, pitch_int))

# ---- verdict ----
final_grav = body_frame(np.array([0.0, 0.0, -1.0]), data.qpos[3:7].copy())
upright = final_grav[2] < -0.9
drift = float(np.linalg.norm(data.qpos[:2]))
rot_err = abs(pitch_int + 2.0 * np.pi)
print("\n---- sim2sim result (MuJoCo) ----")
print(f"pitch integral : {pitch_int:+.3f} rad (target -6.283, err {rot_err:.3f})")
print(f"max height     : {max_h:.3f} m | peak backward rate {-max_wy:.1f} rad/s")
print(f"final height   : {data.qpos[2]:.3f} m | upright: {upright} (g_z {final_grav[2]:+.2f})")
print(f"xy drift       : {drift:.3f} m")
print(f"landing impact : {peak_impact:.0f} N ({peak_impact/147:.1f}x body weight)")
print("t, h, w_y, pitch_int:")
for row in log:
    print(f"  {row[0]:.1f}s  h={row[1]:.3f}  wy={row[2]:+6.2f}  pitch={row[3]:+.3f}")
verdict = rot_err < 0.6 and upright
print(f"\nVERDICT: {'PASS - backflip reproduced in MuJoCo' if verdict else 'FAIL - policy did not transfer'}")

if args.video and frames:
    import imageio
    imageio.mimsave(args.video, frames, fps=50)
    print(f"video -> {args.video}")
