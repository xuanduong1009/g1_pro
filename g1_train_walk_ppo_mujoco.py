from __future__ import annotations
import os
import time
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, Callable, List

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed


# =============================
# Math utils
# =============================
def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float64)

def gravity_in_body(data: mujoco.MjData, model: mujoco.MjModel, body_name: str = "pelvis") -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    R = quat_to_mat(data.xquat[bid])
    return (R @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

def forward_velocity_along_body_x(data: mujoco.MjData, model: mujoco.MjModel, body_name: str = "pelvis") -> float:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    v_world = data.cvel[bid, 3:]
    R_bw = data.xmat[bid].reshape(3, 3)
    x_axis_world = R_bw[:, 0]
    return float(np.dot(v_world, x_axis_world))


# =============================
# Curriculum
# =============================
@dataclass
class Curriculum:
    stand_steps: int = 300_000
    walk_steps: int  = 1_200_000
    run_steps:  int  = 1_500_000
    walk_speed: float = 0.30
    run_speed:  float = 0.80

    def phase(self, global_step: int) -> str:
        if global_step < self.stand_steps:
            return "stand"
        elif global_step < self.stand_steps + self.walk_steps:
            return "walk"
        else:
            return "run"

    def cmd_vx(self, global_step: int) -> float:
        ph = self.phase(global_step)
        if ph == "stand":
            return 0.0
        if ph == "walk":
            t = (global_step - self.stand_steps) / max(1, self.walk_steps)
            return float(min(1.0, t/0.3) * self.walk_speed)
        t = (global_step - self.stand_steps - self.walk_steps) / max(1, self.run_steps)
        return float(self.walk_speed + min(1.0, t/0.4) * (self.run_speed - self.walk_speed))


# =============================
# MuJoCo Env
# =============================
class G1MujocoEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str,
        frame_skip: int = 10,
        torque_limit: float = 60.0,
        action_scale: float = 1.0,
        obs_scales: Dict[str, float] | None = None,
        seed: Optional[int] = None,
        curriculum: Optional[Curriculum] = None,
        alive_bonus: float = 1.0,
        tilt_limit_deg: float = 35.0,
        min_base_z: float = 0.60,
        control_hz: int = 25,
        cmd_noise: float = 0.05,
    ):
        super().__init__()
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random.default_rng()

        # --- load model/data ---
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # --- core params ---
        self.frame_skip = int(frame_skip)
        self.torque_limit = float(torque_limit)
        self.action_scale = float(action_scale)
        self.alive_bonus  = float(alive_bonus)
        self.tilt_limit_rad = math.radians(tilt_limit_deg)
        self.min_base_z   = float(min_base_z)
        self.control_hz   = int(control_hz)
        self.curriculum   = curriculum or Curriculum()
        self.global_step  = 0
        self.cmd_noise    = float(cmd_noise)

        # ---------- robust actuator mapping ----------
        self.nu = int(self.model.nu)
        self._no_actuator_compat = False
        # Compat: nếu XML hiện tại không có actuator nhưng model cũ đòi obs=58 => nu=27
        if self.nu == 0:
            self._no_actuator_compat = True
            self.nu = 27  # 58 = 3 + 2*nu + 1  => nu = 27

        # map joint/dof theo thứ tự actuator nếu có; nếu không có, mảng rỗng
        self._act_joint_id  = np.full(self.nu, -1, dtype=int)
        self._act_qpos_adr  = np.full(self.nu, -1, dtype=int)
        self._act_dof_adr   = np.full(self.nu, -1, dtype=int)
        if not self._no_actuator_compat:
            for a in range(self.model.nu):
                try:
                    j_id = int(self.model.actuator_trnid[a, 0])
                except Exception:
                    j_id = -1
                if 0 <= j_id < self.model.njnt:
                    self._act_joint_id[a] = j_id
                    try:
                        self._act_qpos_adr[a] = int(self.model.jnt_qposadr[j_id])
                    except Exception:
                        self._act_qpos_adr[a] = -1
                    try:
                        self._act_dof_adr[a] = int(self.model.jnt_dofadr[j_id])
                    except Exception:
                        self._act_dof_adr[a] = -1

        # --- obs/action spaces ---
        self.obs_scales = obs_scales or {"qpos": 1.0, "qvel": 0.25}
        obs_dim = 3 + self.nu + self.nu + 1  # phải ra 58 khi nu=27
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float32)

        # --- defaults for reset ---
        self.default_qpos = self.model.key_qpos.copy() if self.model.nkey > 0 else self.model.qpos0.copy()
        self.default_qvel = np.zeros(self.model.nv, dtype=np.float64)

        # --- arms & feet indices ---
        self.arm_qpos_idx: np.ndarray = self._find_arm_qpos_indices()
        self.arm_act_ids:  np.ndarray = self._find_arm_actuator_indices()
        self.arm_hold_k: float = 5.0   # PD mềm giữ tay về 0
        self.arm_hold_d: float = 0.2

        self.foot_body_ids: Tuple[Optional[int], Optional[int]] = self._find_feet_body_ids()

        # prev action
        self._prev_action: Optional[np.ndarray] = None

        self._viewer = None

    # ---------- helpers: indices ----------
    def _find_arm_qpos_indices(self) -> np.ndarray:
        candidates = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        idx: List[int] = []
        for nm in candidates:
            try:
                j_id = self.model.name2id(nm, "joint")
                qadr = int(self.model.jnt_qposadr[j_id])
                if 0 <= qadr < self.model.nq:
                    idx.append(qadr)
            except Exception:
                pass
        return np.array(idx, dtype=int)

    def _find_arm_actuator_indices(self) -> np.ndarray:
        if self.model.nu == 0:
            return np.array([], dtype=int)
        if self.arm_qpos_idx.size == 0:
            return np.array([], dtype=int)
        arm_qset = set(self.arm_qpos_idx.tolist())
        arm_act: List[int] = []
        for a in range(self.model.nu):
            j_id = int(self.model.actuator_trnid[a, 0])
            qadr = int(self.model.jnt_qposadr[j_id])
            if qadr in arm_qset:
                arm_act.append(a)
        return np.array(arm_act, dtype=int)

    def _find_feet_body_ids(self) -> Tuple[Optional[int], Optional[int]]:
        cand_L = ["left_ankle_roll_link", "left_ankle_pitch_link", "left_knee_link", "left_foot", "foot_left"]
        cand_R = ["right_ankle_roll_link", "right_ankle_pitch_link", "right_knee_link", "right_foot", "foot_right"]
        def get_id(names: List[str]) -> Optional[int]:
            for nm in names:
                try:
                    return self.model.name2id(nm, "body")
                except Exception:
                    continue
            return None
        return get_id(cand_L), get_id(cand_R)

    # ---------- obs ----------
    def _get_obs(self, cmd_vx: float) -> np.ndarray:
        g_b = gravity_in_body(self.data, self.model, "pelvis")  # (3,)

        qpos_act = np.zeros(self.nu, dtype=np.float32)
        qvel_act = np.zeros(self.nu, dtype=np.float32)
        if not self._no_actuator_compat:
            # lấy theo thứ tự actuator thật
            for a in range(min(self.nu, self.model.nu)):
                qadr = self._act_qpos_adr[a]
                dadr = self._act_dof_adr[a]
                if 0 <= qadr < self.model.nq:
                    qpos_act[a] = float(self.data.qpos[qadr])
                if 0 <= dadr < self.model.nv:
                    qvel_act[a] = float(self.data.qvel[dadr])
        # nếu compat (nu=27 nhưng model.nu=0), giữ mảng 0s

        qpos_s = self.obs_scales.get("qpos", 1.0) * qpos_act
        qvel_s = self.obs_scales.get("qvel", 0.25) * qvel_act

        return np.concatenate([
            g_b.astype(np.float32),
            qpos_s.astype(np.float32),
            qvel_s.astype(np.float32),
            np.array([cmd_vx], dtype=np.float32),
        ], dtype=np.float32)

    def compute_posture(self) -> Tuple[float, float, float]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        R_bw = self.data.xmat[bid].reshape(3, 3)
        pitch = math.atan2(-R_bw[2, 0], math.sqrt(R_bw[0, 0]**2 + R_bw[1, 0]**2))
        roll  = math.atan2(R_bw[1, 0], R_bw[0, 0])
        base_z = float(self.data.xpos[bid, 2])
        return roll, pitch, base_z

    def _foot_in_contact(self, side: str) -> bool:
        try:
            bid = self.foot_body_ids[0] if side.upper() == "L" else self.foot_body_ids[1]
            if bid is None:
                return False
            z = float(self.data.xpos[bid, 2])
            return z < (self.min_base_z - 0.20)
        except Exception:
            return False

    def _foot_xy_speed(self, side: str) -> float:
        try:
            bid = self.foot_body_ids[0] if side.upper() == "L" else self.foot_body_ids[1]
            if bid is None:
                return 0.0
            vx, vy = float(self.data.cvel[bid, 3]), float(self.data.cvel[bid, 4])
            return float(np.hypot(vx, vy))
        except Exception:
            return 0.0

    # ---------- step/reset ----------
    def step(self, action: np.ndarray):
        cmd_vx = float(np.clip(self.curriculum.cmd_vx(self.global_step) +
                               self.np_random.normal(0.0, self.cmd_noise), -1.2, 1.5))

        action  = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.nu:
            # pad/cắt action để khớp self.nu
            if action.shape[0] < self.nu:
                action = np.pad(action, (0, self.nu - action.shape[0]))
            else:
                action = action[:self.nu]
        action  = np.clip(action, -1.0, 1.0)
        torques = action * self.action_scale * self.torque_limit

        # KHÓA TAY (chỉ khi có actuator thật)
        if (not self._no_actuator_compat) and self.arm_act_ids.size > 0:
            torques[self.arm_act_ids] = 0.0
            for a_id in self.arm_act_ids:
                j_id = int(self.model.actuator_trnid[a_id, 0])
                qadr = int(self.model.jnt_qposadr[j_id])
                dadr = int(self.model.jnt_dofadr[j_id])
                q   = float(self.data.qpos[qadr]) if 0 <= qadr < self.model.nq else 0.0
                dq  = float(self.data.qvel[dadr]) if 0 <= dadr < self.model.nv else 0.0
                torques[a_id] += - self.arm_hold_k * q - self.arm_hold_d * dq

        # apply nếu có actuator thật; nếu không thì bỏ qua
        if not self._no_actuator_compat and self.model.nu > 0:
            self.data.ctrl[:self.model.nu] = np.clip(
                torques[:self.model.nu], -self.torque_limit, self.torque_limit
            )

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs(cmd_vx)

        vx = forward_velocity_along_body_x(self.data, self.model, "pelvis")
        vel_err = vx - cmd_vx
        roll, pitch, base_z = self.compute_posture()
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        y_lat = float(self.data.xpos[pelvis_id, 1])

        W = dict(vel_fwd=2.0, heave=0.5, posture=0.8, lateral=0.6,
                 arm_still=0.4, feet_slip=0.4, action_rate=0.02, torque=0.002)

        r_vel   = W["vel_fwd"] * math.exp(- (vel_err**2) / 0.25)
        r_heave = W["heave"] * math.exp(-20.0 * max(0.0, self.min_base_z - base_z)**2)
        r_post  = - W["posture"] * (abs(roll) + abs(pitch))
        r_lat   = W["lateral"] * math.exp(-3.0 * abs(y_lat))

        if self.arm_qpos_idx.size > 0:
            arm_q = self.data.qpos[self.arm_qpos_idx]
            r_arm = - W["arm_still"] * float(np.mean(arm_q**2))
        else:
            r_arm = 0.0

        slip = 0.0
        if self._foot_in_contact("L"): slip += self._foot_xy_speed("L")
        if self._foot_in_contact("R"): slip += self._foot_xy_speed("R")
        r_slip = - W["feet_slip"] * float(np.clip(slip, 0.0, 1.0))

        if self._prev_action is None:
            self._prev_action = np.zeros_like(action)
        r_rate = - W["action_rate"] * float(np.mean((action - self._prev_action)**2))
        self._prev_action = action.copy()

        r_torque = - W["torque"] * float(np.mean((torques / max(1e-6, self.torque_limit))**2))

        reward = float(r_vel + r_heave + r_post + r_lat + r_arm + r_slip + r_rate + r_torque + self.alive_bonus)

        fallen   = base_z < (self.min_base_z - 0.05)
        too_tilt = (abs(roll) > self.tilt_limit_rad) or (abs(pitch) > self.tilt_limit_rad)
        too_wide = abs(y_lat) > 0.35
        terminated = bool(fallen or too_tilt or too_wide
                          or not np.isfinite(self.data.qpos).all()
                          or not np.isfinite(self.data.qvel).all())
        if terminated:
            reward -= 2.0
        truncated = False

        info: Dict[str, Any] = dict(
            vx=float(vx), cmd_vx=float(cmd_vx), r_vel=float(r_vel),
            r_heave=float(r_heave), r_posture=float(r_post), r_lat=float(r_lat),
            r_arm=float(r_arm), r_slip=float(r_slip), r_action_rate=float(r_rate),
            r_torque=float(r_torque), y_lat=float(y_lat), base_z=float(base_z),
            roll=float(roll), pitch=float(pitch)
        )

        self.global_step += 1
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        qpos = self.default_qpos.copy()
        qvel = self.default_qvel.copy()
        qpos += 0.01 * (self.np_random.random(self.model.nq) - 0.5)
        qvel += 0.01 * (self.np_random.random(self.model.nv) - 0.5)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        self._prev_action = np.zeros(self.nu, dtype=np.float32)

        cmd_vx = self.curriculum.cmd_vx(self.global_step)
        obs = self._get_obs(cmd_vx)
        return obs, {}

    # ---------- render ----------
    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def close(self):
        if self._viewer is not None:
            try: self._viewer.close()
            except Exception: pass
            self._viewer = None


# =============================
# Checkpoint callback
# =============================
class CheckpointCallback(BaseCallback):
    def __init__(self, save_dir: str, save_freq_steps: int = 100_000, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq_steps
        os.makedirs(self.save_dir, exist_ok=True)
    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_dir, f"ppo_g1_{self.num_timesteps}_steps")
            self.model.save(path)
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(os.path.join(self.save_dir, f"vecnorm_{self.num_timesteps}.pkl"))
            if self.verbose:
                print(f"[Checkpoint] Saved at {path}.zip")
        return True


# =============================
# Vec env factory
# =============================
def make_env_factory(
    xml_path: str, seed: int, rank: int,
    frame_skip: int, torque_limit: float, action_scale: float,
    curriculum: Curriculum, tilt_limit_deg: float, min_base_z: float,
    control_hz: int, cmd_noise: float,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        return G1MujocoEnv(
            xml_path=xml_path,
            frame_skip=frame_skip,
            torque_limit=torque_limit,
            action_scale=action_scale,
            curriculum=curriculum,
            tilt_limit_deg=tilt_limit_deg,
            min_base_z=min_base_z,
            control_hz=control_hz,
            cmd_noise=cmd_noise,
            seed=seed + rank,
        )
    return _init


# =============================
# Train / Play entry
# =============================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=str, required=True)
    p.add_argument("--logdir", type=str, default="./runs_g1_mj")
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--frame-skip", type=int, default=10)
    p.add_argument("--torque", type=float, default=60.0)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--stand-steps", type=int, default=300_000)
    p.add_argument("--walk-steps", type=int, default=1_200_000)
    p.add_argument("--run-steps",  type=int, default=1_500_000)
    p.add_argument("--walk-speed", type=float, default=0.30)
    p.add_argument("--run-speed",  type=float, default=0.80)
    p.add_argument("--tilt-limit-deg", type=float, default=35.0)
    p.add_argument("--min-base-z", type=float, default=0.60)
    p.add_argument("--cmd-noise", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--no-subproc", action="store_true")
    p.add_argument("--resume", action="store_true")

    # PPO
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # play
    p.add_argument("--play", action="store_true")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--realtime", action="store_true")

    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    ckpt_dir = os.path.join(args.logdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    curriculum = Curriculum(
        stand_steps=args.stand_steps,
        walk_steps=args.walk_steps,
        run_steps=args.run_steps,
        walk_speed=args.walk_speed,
        run_speed=args.run_speed,
    )

    if args.play:
        env = make_env_factory(
            xml_path=args.xml, seed=args.seed, rank=0,
            frame_skip=args.frame_skip, torque_limit=args.torque, action_scale=args.action_scale,
            curriculum=curriculum, tilt_limit_deg=args.tilt_limit_deg, min_base_z=args.min_base_z,
            control_hz=25, cmd_noise=0.0,
        )()
        if args.model is None or not os.path.exists(args.model):
            raise FileNotFoundError("--model path not found.")
        model = PPO.load(args.model, device="auto")
        print("[play] Loaded:", args.model)
        env.render()
        obs, _ = env.reset()
        try:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if args.realtime:
                    time.sleep(1.0 / max(1, env.control_hz))
                if terminated or truncated:
                    obs, _ = env.reset()
        except KeyboardInterrupt:
            pass
        finally:
            env.close()
        return

    set_random_seed(args.seed)

    env_fns = [make_env_factory(
        xml_path=args.xml, seed=args.seed, rank=i,
        frame_skip=args.frame_skip, torque_limit=args.torque, action_scale=args.action_scale,
        curriculum=curriculum, tilt_limit_deg=args.tilt_limit_deg, min_base_z=args.min_base_z,
        control_hz=25, cmd_noise=args.cmd_noise,
    ) for i in range(args.num_envs)]

    vec_env = DummyVecEnv(env_fns) if (args.no_subproc or os.name == "nt") else SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecNormalize(vec_env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
        gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range,
        ent_coef=args.ent_coef, vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=os.path.join(args.logdir, "tb"), device="auto",
    )

    if args.resume:
        latest = None
        for f in os.listdir(ckpt_dir):
            if f.endswith(".zip") and f.startswith("ppo_g1_"):
                latest = os.path.join(ckpt_dir, f)
        if latest is not None:
            print("[Resume] Loading checkpoint:", latest)
            model = PPO.load(latest, env=vec_env, device="auto")

    ckpt_cb = CheckpointCallback(save_dir=ckpt_dir, save_freq_steps=100_000)
    model.learn(total_timesteps=args.total_steps, callback=ckpt_cb, progress_bar=True)

    latest_path = os.path.join(ckpt_dir, "ppo_g1_latest")
    model.save(latest_path)
    vec_env.save(os.path.join(args.logdir, "vecnorm_latest.pkl"))
    print("[Done] Saved:", latest_path + ".zip")


if __name__ == "__main__":
    main()
