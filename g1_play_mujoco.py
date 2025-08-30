import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import mujoco.viewer

from g1_train_walk_ppo_mujoco import G1MujocoEnv, Curriculum


def make_env(args):
    def _init():
        # Nếu không ép tốc độ: dùng curriculum thường với walk/run speed
        curriculum = None
        if args.force_speed is None:
            curriculum = Curriculum(
                stand_steps=0,
                walk_steps=1_000_000,
                run_steps=1_000_000,
                walk_speed=args.walk_speed,
                run_speed=args.run_speed,
            )
        env = G1MujocoEnv(
            xml_path=args.xml,
            frame_skip=args.frame_skip,
            torque_limit=args.torque,
            action_scale=args.action_scale,
            curriculum=curriculum,
            tilt_limit_deg=args.tilt_limit_deg,
            min_base_z=args.min_base_z,
            cmd_noise=args.cmd_noise,
        )
        # Nếu ép tốc độ: gắn curriculum cố định
        if args.force_speed is not None:
            class _FixedSpeed:
                def cmd_vx(self, global_step: int) -> float:
                    return float(args.force_speed)
            env.curriculum = _FixedSpeed()
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True, help="Path to MuJoCo XML")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO .zip")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to VecNormalize .pkl (optional)")
    parser.add_argument("--episodes", type=int, default=3)

    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--torque", type=float, default=40.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--tilt-limit-deg", type=float, default=25.0)
    parser.add_argument("--min-base-z", type=float, default=0.65)

    # Tham số curriculum/play
    parser.add_argument("--walk-speed", type=float, default=0.30, help="Walk speed khi không ép tốc độ")
    parser.add_argument("--run-speed", type=float, default=0.80, help="Run speed khi không ép tốc độ")
    parser.add_argument("--force-phase", type=str, choices=["stand", "walk", "run"], default=None,
                        help="(Không cần nếu đã --force-speed)")
    parser.add_argument("--cmd-noise", type=float, default=0.0, help="Noise cho lệnh tốc độ khi play")
    parser.add_argument("--force-speed", type=float, default=None, help="Ép tốc độ cố định (m/s)")

    parser.add_argument("--realtime", action="store_true", help="Sleep theo control_hz khi play")
    parser.add_argument("--deterministic", action="store_true", help="Predict deterministic")

    args = parser.parse_args()

    # ---------- Env + VecNormalize (fallback nếu lệch shape) ----------
    base_env = DummyVecEnv([make_env(args)])
    env = base_env
    if args.vecnorm:
        try:
            print(f"[Info] Loading VecNormalize: {args.vecnorm}")
            loaded = VecNormalize.load(args.vecnorm, base_env)
            if loaded.observation_space.shape == base_env.observation_space.shape:
                loaded.training = False
                loaded.norm_reward = False
                env = loaded
                print("[Info] VecNormalize loaded & attached.")
            else:
                print("[Warn] VecNormalize obs shape mismatch → dùng base_env không normalize.")
        except Exception as e:
            print(f"[Warn] VecNormalize load failed ({e}) → dùng base_env.")

    # ---------- Model ----------
    print(f"[Info] Loading model: {args.model}")
    model = PPO.load(args.model, device="auto")

    # ---------- Viewer ----------
    raw_env = base_env.envs[0]  # env thật để sync viewer & lấy control_hz
    viewer = mujoco.viewer.launch_passive(raw_env.model, raw_env.data)
    print("[Viewer] Opened via mujoco.viewer.launch_passive")

    if args.force_phase and args.force_speed is None:
        print("[Warn] --force-phase được set nhưng play không chạy theo phase train. Khuyên dùng --force-speed.")

    # ---------- Rollout ----------
    for ep in range(args.episodes):
        obs = env.reset()
        if isinstance(obs, tuple):  # một số wrapper trả (obs, info)
            obs = obs[0]

        ep_ret = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = env.step(action)
            ep_ret += float(rewards[0]) if np.ndim(rewards) > 0 else float(rewards)

            if viewer.is_running():
                viewer.sync()
            else:
                print("[Viewer] Closed by user.")
                done = True
                break

            if args.realtime:
                time.sleep(1.0 / max(1, raw_env.control_hz))

            done = bool(dones[0]) if np.ndim(dones) > 0 else bool(dones)

        print(f"[Episode {ep+1}/{args.episodes}] return = {ep_ret:.2f}")

    print("[Done] Play finished.")


if __name__ == "__main__":
    main()
