import bpy, numpy as np
from mathutils import Quaternion
import os

# ---------- CONFIG ----------
FILENAME = "huskyb_switch_to_walk_simple.npz"
OUT = "/home/arjun/Documents/sslab_isaaclab/source/sslab_extensions/simulation/blender/huskyb"

ARMATURE = "root"                         # armature object name
BASE_BONE = "root.bone"                   # root/base pose bone name
JOINT_NAMES = [
    "bl_hx.revolute.bone", "br_hx.revolute.bone", "fl_hx.revolute.bone", "fr_hx.revolute.bone",
    "bl_hy.revolute.bone", "br_hy.revolute.bone", "fl_hy.revolute.bone", "fr_hy.revolute.bone",
    "bl_kn.revolute.bone", "br_kn.revolute.bone", "fl_kn.revolute.bone", "fr_kn.revolute.bone",
]
USD_NAMES = [
    "bl_hx", "br_hx", "fl_hx", "fr_hx", 
    "bl_hy", "br_hy", "fl_hy", "fr_hy",
    "bl_kn", "br_kn", "fl_kn", "fr_kn",
]
# hip=X, thigh&calf=Y
COPY_ID = [1,1,1,1, 1,1,1,1, 1,1,1,1]

# Bodies you want tracked (example: feet; add others as needed)
BODY_BONES = [
    "bl_fe.fixed.bone", "br_fe.fixed.bone", "fl_fe.fixed.bone", "fr_fe.fixed.bone",
]
USD_BODY_NAMES = [
    "bl_foot", "br_foot", "fl_foot", "fr_foot",
]
# ---------------------------------------
os.makedirs(OUT, exist_ok=True)

scn = bpy.context.scene
fps = float(scn.render.fps)
dt = 1.0 / fps
arm = bpy.data.objects[ARMATURE]
pose = arm.pose

frames = list(range(scn.frame_start, scn.frame_end + 1))
N, J, B = len(frames), len(JOINT_NAMES), len(BODY_BONES)

base_pos  = np.zeros((N,3), dtype=np.float64)
base_quat = np.zeros((N,4), dtype=np.float64)  # wxyz
qpos      = np.zeros((N,J), dtype=np.float64)

# Bodies (world)
body_pos  = np.zeros((N,B,3), dtype=np.float64)
body_quat = np.zeros((N,B,4), dtype=np.float64)

# ---- sample animation ----
for i, f in enumerate(frames):
    scn.frame_set(f)

    # base world transform
    M = arm.matrix_world @ pose.bones[BASE_BONE].matrix
    t = M.to_translation()
    q = M.to_quaternion()
    base_pos[i]  = (t.x/100.0, t.y/100.0, t.z/100.0)    # convert cm->m if your rig is in cm
    base_quat[i] = (q.w, q.x, q.y, q.z)

    # joints
    for j, name in enumerate(JOINT_NAMES):
        e = pose.bones[name].rotation_euler
        qpos[i, j] = (e.x, e.y, e.z)[COPY_ID[j]]

    # bodies (e.g., feet) world pose
    for b, bname in enumerate(BODY_BONES):
        Mb = arm.matrix_world @ pose.bones[bname].matrix
        tb = Mb.to_translation()
        qb = Mb.to_quaternion()
        body_pos[i, b]  = (tb.x/100.0, tb.y/100.0, tb.z/100.0)
        body_quat[i, b] = (qb.w, qb.x, qb.y, qb.z)

# ---- velocities ----
# base linear velocity (m/s)
base_lin_vel = np.gradient(base_pos, dt, axis=0)

# base angular velocity from quaternion deltas
omega = np.zeros((N,3), dtype=np.float64)
if N >= 2:
    for k in range(N-1):
        qk = Quaternion((base_quat[k,0],  *base_quat[k,1:4]))
        qn = Quaternion((base_quat[k+1,0],*base_quat[k+1,1:4]))
        dq = qn @ qk.conjugated()
        axis, ang = dq.to_axis_angle()
        if ang > np.pi: ang -= 2*np.pi
        omega[k] = np.array(axis) * (ang / dt)
    omega[-1] = omega[-2]
base_ang_vel = omega

# joint velocities (rad/s)
qvel = np.gradient(qpos, dt, axis=0)

# body linear velocity (m/s)
body_lin_vel = np.gradient(body_pos, dt, axis=0)  # (N,B,3)

# body angular velocity (rad/s) from quaternion deltas
body_ang_vel = np.zeros((N,B,3), dtype=np.float64)
if N >= 2:
    for b in range(B):
        for k in range(N-1):
            qk = Quaternion((body_quat[k,b,0],  *body_quat[k,b,1:4]))
            qn = Quaternion((body_quat[k+1,b,0],*body_quat[k,b,1:4]))
            dq = qn @ qk.conjugated()
            axis, ang = dq.to_axis_angle()
            if ang > np.pi: ang -= 2*np.pi
            body_ang_vel[k,b] = np.array(axis) * (ang / dt)
        body_ang_vel[-1,b] = body_ang_vel[-2,b]

time = np.arange(N, dtype=np.float64) * dt

# ---- save ----
np.savez_compressed(
    os.path.join(OUT, FILENAME),
    time=time,
    base_pos_w=base_pos.astype(np.float64),
    base_quat_w=base_quat.astype(np.float64),         # (w,x,y,z)
    base_lin_vel_w=base_lin_vel.astype(np.float64),
    base_ang_vel_w=base_ang_vel.astype(np.float64),
    qpos=qpos.astype(np.float64),
    qvel=qvel.astype(np.float64),
    joint_names=np.array(USD_NAMES),

    # bodies:
    body_names=np.array(USD_BODY_NAMES),
    body_pos_w=body_pos.astype(np.float64),           # (N,B,3)
    body_quat_w=body_quat.astype(np.float64),         # (N,B,4)
    body_lin_vel_w=body_lin_vel.astype(np.float64),   # (N,B,3)
    body_ang_vel_w=body_ang_vel.astype(np.float64),   # (N,B,3)

    root_name=np.array(["base"]),
)

print(f"[OK] Saved {OUT}")
print("qpos:", qpos.shape, "qvel:", qvel.shape)
print("base_pos:", base_pos.shape, "base_quat:", base_quat.shape)
print("base_lin_vel:", base_lin_vel.shape, "base_ang_vel:", base_ang_vel.shape)
print("body_pos:", body_pos.shape, "body_lin_vel:", body_lin_vel.shape)
