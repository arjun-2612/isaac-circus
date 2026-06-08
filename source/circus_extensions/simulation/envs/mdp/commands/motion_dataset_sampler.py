from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .motion_dataset_sampler_cfg import MotionDatasetSamplerCfg


# ----------------------------- utilities -----------------------------
import torch

def transform_to_se2_relative_frame(robot_data: dict[str, torch.Tensor], root_idx: int = 0) -> dict[str, torch.Tensor]:
    """
    Re-base trajectory to the root link's initial SE(2) frame (x0,y0,yaw0) at t=0.
    Inputs (torch tensors, leading T):
      quat_w:    [T,4]  (w,x,y,z) world
      pos_w:     [T,3]  xyz world
      lin_vel_w: [T,3]  m/s world
      ang_vel_w: [T,3]  rad/s world
    Returns a copy with the same keys, expressed in the SE(2)-relative frame.
    """
    # pull tensors (keep dtype/device)
    quat_w: torch.Tensor = robot_data["quat_w"]          # [T,4] wxyz
    pos_w:  torch.Tensor = robot_data["pos_w"]           # [T,3]
    lv_w:   torch.Tensor = robot_data["lin_vel_w"]       # [T,3]
    av_w:   torch.Tensor = robot_data["ang_vel_w"]       # [T,3]

    # helpers
    def quat_inv(q):  # q: (...,4) wxyz
        w,x,y,z = q.unbind(-1)
        nq2 = (w*w + x*x + y*y + z*z).clamp_min(torch.finfo(q.dtype).tiny)
        return torch.stack([ w/nq2, -x/nq2, -y/nq2, -z/nq2 ], dim=-1)

    def quat_mul(q1, q2):  # Hamilton product, both (...,4) wxyz
        w1,x1,y1,z1 = q1.unbind(-1)
        w2,x2,y2,z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    def yaw_from_quat_wxyz(q):  # q: (4,)
        w,x,y,z = q
        # standard ZYX yaw extraction
        return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def yaw_quat(yaw):  # returns (w,x,y,z) for pure yaw
        hy = 0.5 * yaw
        return torch.stack([torch.cos(hy), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(hy)], dim=-1)

    def Rz_inv_apply(vxy, c, s):  # rotate XY by Rz(-yaw)
        x = vxy[..., 0]
        y = vxy[..., 1]
        xr =  c * x + s * y
        yr = -s * x + c * y
        return torch.stack([xr, yr], dim=-1)

    # initial root pose (t=0)
    q0 = quat_w[0]                # (4,)
    p0 = pos_w[0]                 # (3,)
    yaw0 = yaw_from_quat_wxyz(q0) # scalar tensor
    cy, sy = torch.cos(yaw0), torch.sin(yaw0)

    # remove initial yaw from quats
    q0_yaw = yaw_quat(yaw0)                 # (4,)
    q0_yaw_inv = quat_inv(q0_yaw).view(1,4) # (1,4)
    rel_quat = quat_mul(q0_yaw_inv.expand_as(quat_w), quat_w)  # [T,4]

    # positions: translate by -p0 in XY then rotate by Rz(-yaw0)
    rel_pos = pos_w.clone()
    dxy = pos_w[:, :2] - p0[:2]
    rel_pos[:, :2] = Rz_inv_apply(dxy, cy, sy)  # z unchanged

    # velocities: rotate world -> yaw0-relative (XY only)
    rel_lv = lv_w.clone()
    rel_lv[:, :2] = Rz_inv_apply(lv_w[:, :2], cy, sy)

    rel_av = av_w.clone()
    rel_av[:, :2] = Rz_inv_apply(av_w[:, :2], cy, sy)  # z stays same under Rz

    out = dict(robot_data)
    out["quat_w"]    = rel_quat
    out["pos_w"]     = rel_pos
    out["lin_vel_w"] = rel_lv
    out["ang_vel_w"] = rel_av

    # ===================== Bodies (optional) =====================
    # Positions/orientations/velocities of object links in world
    body_pos_w = robot_data.get("body_pos_w")                                   # [T,B,3]
    body_quat_w = robot_data.get("body_quat_w")                                 # [T,B,4]
    body_lin_vel_w = robot_data.get("body_lin_vel_w")                           # [T,B,3]
    body_ang_vel_w = robot_data.get("body_ang_vel_w")                           # [T,B,3]

    rel_bpos = body_pos_w.clone()
    dxy_b = body_pos_w[..., :2] - p0[:2]  # broadcasts over B
    rel_bpos[..., :2] = Rz_inv_apply(dxy_b, cy, sy)
    out["body_pos_w"] = rel_bpos

    # left-multiply every body quat by q0_yaw_inv
    # reshape q0_yaw_inv to [1,1,4] to broadcast over [T,B,4]
    q0i_tb = q0_yaw_inv.view(1, 1, 4).expand(body_quat_w.shape[0], body_quat_w.shape[1], 4)
    out["body_quat_w"] = quat_mul(q0i_tb, body_quat_w)

    rel_blv = body_lin_vel_w.clone()
    rel_blv[..., :2] = Rz_inv_apply(body_lin_vel_w[..., :2], cy, sy)
    out["body_lin_vel_w"] = rel_blv

    rel_bav = body_ang_vel_w.clone()
    rel_bav[..., :2] = Rz_inv_apply(body_ang_vel_w[..., :2], cy, sy)
    out["body_ang_vel_w"] = rel_bav

    return out


# ----------------------------- main sampler -----------------------------

class MotionDatasetSampler(CommandTerm):
    """Single-clip reference sampler: every env tracks the same reference with its own phase."""

    cfg: MotionDatasetSamplerCfg

    def __init__(self, cfg: MotionDatasetSamplerCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # -- robot / device
        self.robot: Articulation = env.scene[cfg.robot_asset_name]

        # -- load npz (to torch) -----------------
        npz = dict(np.load(cfg.data_dir))
        def t(x): return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        # Required keys (T-first)
        self._qpos = t(npz["qpos"])                       # [T, J]
        self._quat_w = t(npz["base_quat_w"])              # [T, 4] (w,x,y,z)
        self._pos_w = t(npz["base_pos_w"])                # [T, 3]
        self._lin_vel_w = t(npz["base_lin_vel_w"])        # [T, 3]
        self._ang_vel_w = t(npz["base_ang_vel_w"])        # [T, 3]
        self._body_pos_w = t(npz["body_pos_w"])           # [T, B, 3]
        self._body_quat_w = t(npz["body_quat_w"])         # [T, B, 4]
        self._body_lin_vel_w = t(npz["body_lin_vel_w"])   # [T, B, 3]
        self._body_ang_vel_w = t(npz["body_ang_vel_w"])   # [T, B, 3]

        self._time = t(npz["time"]).flatten()             # [T]
        self._traj_time_step = float(torch.median(self._time[1:] - self._time[:-1]).item())

        # last-frame velocities to zero (nicer edges)
        self._lin_vel_w[-1] = 0.0
        self._ang_vel_w[-1] = 0.0

        # -- SE(2) relative transform (once) ------
        # (root link index 0 by default; change if different)
        self._root_idx = getattr(cfg, "root_body_index", 0)
        self._robot_data = {
            "quat_w": self._quat_w,
            "pos_w": self._pos_w,
            "lin_vel_w": self._lin_vel_w,
            "ang_vel_w": self._ang_vel_w,
            "body_pos_w": self._body_pos_w,
            "body_quat_w": self._body_quat_w,
            "body_lin_vel_w": self._body_lin_vel_w,
            "body_ang_vel_w": self._body_ang_vel_w,
        }
        self._robot_data = transform_to_se2_relative_frame(self._robot_data, self._root_idx)
        self._quat_w = self._robot_data["quat_w"]
        self._pos_w = self._robot_data["pos_w"]
        self._lin_vel_w = self._robot_data["lin_vel_w"]
        self._ang_vel_w = self._robot_data["ang_vel_w"]
        self._body_pos_w = self._robot_data["body_pos_w"]
        self._body_quat_w = self._robot_data["body_quat_w"]
        self._body_lin_vel_w = self._robot_data["body_lin_vel_w"]
        self._body_ang_vel_w = self._robot_data["body_ang_vel_w"]

        # -- lengths / duration -------------------
        self._dataset_length = int(self._time.shape[0])   # T
        self._task_length_s = float(self._time[-1].item())  # duration in seconds

        # --- control timing (policy/manager rate) ---
        self._control_dt = float(env.cfg.sim.dt * env.cfg.decimation)  # e.g., 0.02 for 50 Hz

        # per-env reference time (seconds)
        self._t = torch.zeros(self.num_envs, device=self.device)

        # frame mix buffers computed each control tick
        self._i0 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._i1 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._alpha = torch.zeros(self.num_envs, 1, device=self.device)  # [N,1]

        # a tiny helper to gather & mix frames quickly
        def _gather_lin(vecT, i0, i1, alpha):
            """vecT: [T,...]  ->  [N,...] linear mix"""
            v0 = vecT.index_select(0, i0)
            v1 = vecT.index_select(0, i1)
            return v0 * (1 - alpha) + v1 * alpha

        def _slerp(qT, i0, i1, alpha, eps=1e-8):
            q0 = qT.index_select(0, i0)
            q1 = qT.index_select(0, i1)
            # enforce shortest path
            dot = (q0 * q1).sum(-1, keepdim=True)
            q1 = torch.where(dot < 0, -q1, q1)
            dot = dot.clamp(-1.0, 1.0)
            omega = torch.acos(dot)
            small = (omega < 1e-4)

            # linear blend for small angles
            q_lin = (1 - alpha) * q0 + alpha * q1

            # slerp for normal cases
            sin_omega = torch.sin(omega).clamp_min(eps)
            w0 = torch.sin((1 - alpha) * omega) / sin_omega
            w1 = torch.sin(alpha * omega) / sin_omega
            q_slerp = w0 * q0 + w1 * q1

            q = torch.where(small, q_lin, q_slerp)
            q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
            return q

        def _slerp_TB(qTB, i0, i1, alpha, eps=1e-8):
            q0 = qTB.index_select(0, i0)  # [N,B,4]
            q1 = qTB.index_select(0, i1)  # [N,B,4]
            dot = (q0 * q1).sum(-1, keepdim=True)
            q1 = torch.where(dot < 0, -q1, q1)
            dot = dot.clamp(-1.0, 1.0)
            omega = torch.acos(dot)
            small = (omega < 1e-4)

            a = alpha.view(-1,1,1)
            q_lin = (1 - a) * q0 + a * q1

            sin_omega = torch.sin(omega).clamp_min(eps)
            w0 = torch.sin((1 - a) * omega) / sin_omega
            w1 = torch.sin(a * omega) / sin_omega
            q_slerp = w0 * q0 + w1 * q1

            q = torch.where(small, q_lin, q_slerp)
            q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
            return q

        self._gather_lin = _gather_lin
        self._slerp = _slerp
        self._slerp_TB = _slerp_TB

        # -- buffers (per-env) --------------------
        self._task_phase = torch.zeros(self.num_envs, device=self.device)      # [N], 0..1
        self._time_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # [N]
        self._task_completion_ratio = torch.zeros(self.num_envs, device=self.device)
        self._reached_end_of_task = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # -- metrics buffers ----------------------
        for k in cfg.tracking_term_tolerance.keys():
            self.metrics[k + "_error_norm"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[k + "_score"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["similarity_metric"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["task_completion_ratio"] = torch.zeros_like(self._task_completion_ratio)

        # curriculum flags
        self._curriculum_update = {
            "difficulty_increase_envs": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "difficulty_decrease_envs": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

        # precompute extras
        self._init_extras()

        print(self)

    """
    Properties
    """

    @property
    def task_completion_ratio(self) -> torch.Tensor:
        return self._task_completion_ratio

    @property
    def task_phase(self) -> torch.Tensor:
        return self._task_phase

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._gather_lin(self._qpos, self._i0, self._i1, self._alpha)  # [N,J]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._gather_lin(self._joint_vel_traj, self._i0, self._i1, self._alpha)  # [N,J]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._gather_lin(self._pos_w, self._i0, self._i1, self._alpha)  # [N,3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._slerp(self._quat_w, self._i0, self._i1, self._alpha)  # [N,4]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._gather_lin(self._lin_vel_w, self._i0, self._i1, self._alpha)  # [N,3]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._gather_lin(self._ang_vel_w, self._i0, self._i1, self._alpha)  # [N,3]

    @property
    def projected_gravity(self) -> torch.Tensor:
        return self._gather_lin(self._projected_gravity_traj, self._i0, self._i1, self._alpha)  # [N,3]

    @property
    def root_lin_vel_b(self) -> torch.Tensor:
        return self._gather_lin(self._root_lin_vel_b_traj, self._i0, self._i1, self._alpha)  # [N,3]

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        return self._gather_lin(self._root_ang_vel_b_traj, self._i0, self._i1, self._alpha)  # [N,3]

    @property
    def body_pos_w(self) -> torch.Tensor:
        v0 = self._body_pos_w.index_select(0, self._i0)  # [N,B,3]
        v1 = self._body_pos_w.index_select(0, self._i1)  # [N,B,3]
        a = self._alpha.view(-1,1,1)
        return v0 * (1 - a) + v1 * a

    @property
    def root_lin_acc_w(self) -> torch.Tensor:
        return self._gather_lin(self._root_lin_acc_w_traj, self._i0, self._i1, self._alpha)  # [N,3]

    @property
    def root_ang_acc_w(self) -> torch.Tensor:
        return self._gather_lin(self._root_ang_acc_w_traj, self._i0, self._i1, self._alpha)  # [N,3]


    @property
    def command(self):
        pass


    def _init_extras(self):
        T = self._dataset_length
        dt = self._traj_time_step

        # joint vel [T, J]
        self._joint_vel_traj = torch.zeros_like(self._qpos)
        self._joint_vel_traj[1:] = (self._qpos[1:] - self._qpos[:-1]) / dt

        self._root_lin_acc_w_traj = torch.zeros_like(self._lin_vel_w)
        self._root_ang_acc_w_traj = torch.zeros_like(self._ang_vel_w)
        self._root_lin_acc_w_traj[1:-1] = (self._lin_vel_w[1:-1] - self._lin_vel_w[:-2]) / dt
        self._root_ang_acc_w_traj[1:-1] = (self._ang_vel_w[1:-1] - self._ang_vel_w[:-2]) / dt

        # projected gravity / velocities in body frame
        g_w = self.robot.data.GRAVITY_VEC_W[0].to(self.device)  # [3]
        g_w_T = g_w.view(1, 3).expand(T, 3)
        self._projected_gravity_traj = math_utils.quat_rotate_inverse(self._quat_w, g_w_T)  # [T,3]
        self._root_lin_vel_b_traj = math_utils.quat_rotate_inverse(self._quat_w, self._lin_vel_w)
        self._root_ang_vel_b_traj = math_utils.quat_rotate_inverse(self._quat_w, self._ang_vel_w)

        # body positions wrt base in base frame (optional; keep if needed)
        # ref_t_inv = -R(q)^T * p  ; implemented via helper
        # If you need this, math_utils.transform_points supports batched inputs.

        # per-term stds (avoid zero)
        mins = self.cfg.tracking_term_tolerance
        self._root_lin_vel_traj_std = torch.std(self._root_lin_vel_b_traj, dim=0).clamp_min(mins["root_lin_vel"])
        self._root_ang_vel_traj_std = torch.std(self._root_ang_vel_b_traj, dim=0).clamp_min(mins["root_ang_vel"])
        self._projected_gravity_traj_std = torch.std(self._projected_gravity_traj, dim=0).clamp_min(mins["projected_gravity"])
        self._joint_pos_traj_std = torch.std(self._qpos, dim=0).clamp_min(mins["joint_pos"])

    """
    Implementation specific functions.
    """

    def _update_command(self):
        # advance per-env reference time by control_dt
        self._t = torch.minimum(
            self._t + self._control_dt,
            torch.tensor(self._task_length_s + self.cfg.max_extra_time, device=self.device)
        )
        # update phase 0..1 for metrics/gating
        self._task_phase = (self._t / max(self._task_length_s, 1e-6)).clamp(0.0, 1.0)

        # compute i0,i1,alpha for this tick (also sets _time_indices=i0)
        self._compute_frame_mix()

        # reached end of *task* (not counting extra wait)
        self._reached_end_of_task = (self._t >= self._task_length_s)
        
    def _update_metrics(self):
        self.compute_tracking_metrics()
        self.metrics["similarity_metric"] += self.metrics["joint_pos_score"] / self._env.max_episode_length
        self.metrics["task_completion_ratio"] = self._task_completion_ratio.clone()

    def _resample_command(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            # seed time uniformly in [phase_min, phase_max] of the clip
            lo = self.cfg.task_phase_min * self._task_length_s
            hi = self.cfg.task_phase_max * self._task_length_s
            self._t = torch.rand(self.num_envs, device=self.device) * (hi - lo) + lo
            self._task_completion_ratio.zero_()
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            lo = self.cfg.task_phase_min * self._task_length_s
            hi = self.cfg.task_phase_max * self._task_length_s
            self._t[env_ids] = torch.rand(env_ids.numel(), device=self.device) * (hi - lo) + lo
            self._task_completion_ratio[env_ids] = self._reached_end_of_task[env_ids].float()

        self._task_phase = (self._t / max(self._task_length_s, 1e-6)).clamp(0.0, 1.0)
        self._compute_frame_mix()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

    """
    Internal helpers.
    """

    def _compute_frame_mix(self):
        """Compute i0, i1, alpha from current per-env time self._t."""
        T = self._dataset_length
        f = (self._t / self._traj_time_step).clamp(0.0, T - 1 - 1e-8)  # [N]
        i0 = f.floor().to(torch.long)
        i1 = torch.clamp(i0 + 1, max=T - 1)
        alpha = (f - i0).unsqueeze(-1)  # [N,1]

        self._i0.copy_(i0)
        self._i1.copy_(i1)
        self._alpha.copy_(alpha)

        # maintain compatibility with code that uses _time_indices
        self._time_indices = i0

    def compute_tracking_metrics(self):
        # All gathers are already [N,...] from properties above
        err = (self.projected_gravity - self.robot.data.projected_gravity_b) / self._projected_gravity_traj_std
        # print(f"Reference proj: {self.projected_gravity}, Asset proj: {self.robot.data.projected_gravity_b}")
        self.metrics["projected_gravity_error_norm"] = err.norm(dim=-1)

        err = (self.root_lin_vel_b - self.robot.data.root_lin_vel_b) / self._root_lin_vel_traj_std
        self.metrics["root_lin_vel_error_norm"] = err.norm(dim=-1)

        err = (self.root_ang_vel_b - self.robot.data.root_ang_vel_b) / self._root_ang_vel_traj_std
        # print(f"Reference ang vel: {self.root_ang_vel_b}, Asset ang vel: {self.robot.data.root_ang_vel_b}")
        self.metrics["root_ang_vel_error_norm"] = err.norm(dim=-1)

        # print(self.robot.data.applied_torque)

        err = (self.joint_pos - self.robot.data.joint_pos[..., :12]) / self._joint_pos_traj_std
        self.metrics["joint_pos_error_norm"] = err.norm(dim=-1)

        for k in self.cfg.tracking_term_tolerance.keys():
            self.metrics[k + "_score"] = torch.exp(-self.metrics[k + "_error_norm"] / self.cfg.score_std)

    def get_curriculum_info(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        sim = self.metrics["similarity_metric"]
        is_close = (torch.rand_like(sim) < sim)
        self._curriculum_update["difficulty_increase_envs"] = is_close
        self._curriculum_update["difficulty_decrease_envs"] = ~is_close
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        return (
            self._curriculum_update["difficulty_decrease_envs"][env_ids_t],
            self._curriculum_update["difficulty_increase_envs"][env_ids_t],
        )

    def __str__(self) -> str:
        return (
            f"Dataset file: {self.cfg.data_dir}\n"
            f"Duration: {self._task_length_s:.3f}s | Steps: {self._dataset_length} | dt≈{self._traj_time_step:.4f}s\n"
            f"Root index: {self._root_idx} | Env count: {self.num_envs}"
        )
