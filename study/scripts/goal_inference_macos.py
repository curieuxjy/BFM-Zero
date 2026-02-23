"""
실습 5: macOS용 Goal Inference 스크립트

goal_frames_lafan29dof.json에서 목표 프레임을 로드하고,
B(goal_obs)로 z 벡터를 계산한 뒤 롤아웃을 실행합니다.

사용법:
    # GUI 모드:
    uv run python study/scripts/goal_inference_macos.py --model_folder model/

    # Headless 모드:
    uv run python study/scripts/goal_inference_macos.py --model_folder model/ --headless

    # 특정 goal index:
    uv run python study/scripts/goal_inference_macos.py --model_folder model/ --goal_idx 0
"""
import os
import sys

if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "glfw"
else:
    os.environ["MUJOCO_GL"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# MuJoCo 패치 적용
from humanoidverse.simulator.mujoco import mujoco as mujoco_module
from humanoidverse.utils.torch_utils import quat_rotate
import torch

MuJoCoClass = mujoco_module.MuJoCo


def _patched_robot_root_states(self):
    base_quat = self.base_quat
    qvel_tensor = torch.tensor([self.data.qvel[0:6]], device=self.device, dtype=torch.float32)
    return torch.cat([
        torch.tensor([self.data.qpos[0:3]], device=self.device, dtype=torch.float32),
        base_quat,
        qvel_tensor[:, 0:3],
        quat_rotate(base_quat, qvel_tensor[:, 3:6], w_last=True),
    ], dim=-1)


def _patched_all_root_states(self):
    base_quat = self.base_quat
    qvel_tensor = torch.tensor([self.data.qvel[0:6]], device=self.device, dtype=torch.float32)
    return torch.cat([
        torch.tensor([self.data.qpos[0:3]], device=self.device, dtype=torch.float32),
        base_quat,
        qvel_tensor[:, 0:3],
        quat_rotate(base_quat, qvel_tensor[:, 3:6], w_last=True),
    ], dim=-1)


MuJoCoClass.robot_root_states = property(_patched_robot_root_states)
MuJoCoClass.all_root_states = property(_patched_all_root_states)
print("[patch] MuJoCo quat_rotate w_last 패치 적용됨")

import json
import numpy as np
import mujoco
import cv2
import tyro

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.utils.helpers import export_meta_policy_as_onnx, get_backward_observation

import humanoidverse
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent.parent.parent / "humanoidverse"


def main(
    model_folder: Path,
    goal_idx: int = 0,
    headless: bool = False,
    device: str = "cpu",
    num_steps: int = 500,
):
    """
    macOS용 Goal Inference 스크립트

    Args:
        model_folder: 모델 폴더 경로
        goal_idx: goal_frames_lafan29dof.json에서 사용할 goal 인덱스
        headless: GUI 없이 실행
        device: cpu
        num_steps: 시뮬레이션 스텝 수
    """
    model_folder = Path(model_folder)

    # Goal frames 로드
    goal_frames_path = HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "goal_frames_lafan29dof.json"
    if not goal_frames_path.exists():
        print(f"Error: {goal_frames_path} not found")
        return

    with open(goal_frames_path) as f:
        goal_frames = json.load(f)

    print(f"Available goals ({len(goal_frames)}):")
    for i, g in enumerate(goal_frames[:10]):
        marker = " <--" if i == goal_idx else ""
        print(f"  [{i}] Motion: {g.get('motion_name', 'N/A')}, "
              f"ID: {g.get('motion_id', 'N/A')}, "
              f"Frames: {g.get('frames', 'N/A')}{marker}")
    if len(goal_frames) > 10:
        print(f"  ... ({len(goal_frames) - 10} more)")

    if goal_idx >= len(goal_frames):
        print(f"Error: goal_idx {goal_idx} out of range (max {len(goal_frames) - 1})")
        return

    goal = goal_frames[goal_idx]
    motion_id = goal["motion_id"]
    goal_frame = goal["frames"][0] if isinstance(goal["frames"], list) else goal["frames"]
    print(f"\nUsing goal: motion_id={motion_id}, frame={goal_frame}")

    # 모델 로드
    print("Loading model...")
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()

    # 설정 로드
    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    use_root_height_obs = config["env"].get("root_height_obs", False)

    if not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)

    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"++headless={headless}")
    config["env"]["hydra_overrides"].append("simulator=mujoco")
    config["env"]["disable_domain_randomization"] = True
    config["env"]["disable_obs_noise"] = True

    # 환경 빌드
    print("Building environment...")
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(1)
    env = wrapped_env._env

    # Goal z 계산: B(goal_obs)
    print(f"Computing goal z for motion {motion_id}, frame {goal_frame}...")
    env.set_is_evaluating(motion_id)
    obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)

    with torch.no_grad():
        # goal_frame에 해당하는 관측만 사용
        frame_idx = min(goal_frame, obs["state"].shape[0] - 1)
        goal_obs = {k: v[frame_idx:frame_idx+1] for k, v in obs.items()}
        z = model.backward_map(goal_obs)
        z = model.project_z(z)

    print(f"Goal z shape: {z.shape}, ||z||: {z.norm().item():.2f}")

    # 환경 리셋
    wrapped_obs, _ = wrapped_env.reset(to_numpy=False)

    mj_model = env.simulator.model
    mj_data = env.simulator.data

    if not headless:
        print("\n" + "=" * 60)
        print("Goal Inference Viewer")
        print(f"Goal: motion {motion_id}, frame {goal_frame}")
        print("Press 'q' or ESC to exit.")
        print("=" * 60 + "\n")

        mj_model.vis.global_.offwidth = 1280
        mj_model.vis.global_.offheight = 720
        renderer = mujoco.Renderer(mj_model, height=720, width=1280)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 4.0
        cam.azimuth = 90
        cam.elevation = -20
        cam.lookat[:] = [0, 0, 0.8]

        cv2.namedWindow("Goal Inference", cv2.WINDOW_NORMAL)
        step_counter = 0

        while True:
            with torch.no_grad():
                action = model.act(wrapped_obs, z, mean=True)

            wrapped_obs, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)

            renderer.update_scene(mj_data, camera=cam)
            cam.lookat[:] = mj_data.qpos[:3]
            cam.lookat[2] = 0.8
            frame = renderer.render()

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 정보 텍스트 오버레이
            cv2.putText(frame_bgr, f"Goal: M{motion_id} F{goal_frame}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Step: {step_counter}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Goal Inference", frame_bgr)
            step_counter += 1

            if terminated or truncated:
                wrapped_obs, _ = wrapped_env.reset(to_numpy=False)
                step_counter = 0

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q") or key == 27:
                break

        cv2.destroyAllWindows()
        renderer.close()
        print(f"\nViewer closed after {step_counter} steps")

    else:
        print(f"\nRunning goal inference for {num_steps} steps...")
        for step in range(num_steps):
            with torch.no_grad():
                action = model.act(wrapped_obs, z, mean=True)

            wrapped_obs, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)

            if step % 100 == 0:
                pos = mj_data.qpos[:3]
                print(f"  Step {step}/{num_steps}, pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

            if terminated or truncated:
                wrapped_obs, _ = wrapped_env.reset(to_numpy=False)

        print("  Rollout complete")

    print("\nDone!")


if __name__ == "__main__":
    tyro.cli(main)
