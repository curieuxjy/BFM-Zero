"""
macOS용 tracking inference 스크립트

사용법:
    # GUI 모드 (cv2 뷰어 사용, mjpython 불필요):
    uv run python study/scripts/tracking_inference_macos.py --model_folder model/

    # Headless 모드:
    uv run python study/scripts/tracking_inference_macos.py --model_folder model/ --headless

이 스크립트는 메인 소스코드를 수정하지 않고 런타임 패치를 적용합니다.
"""
import os
import sys

# offscreen 렌더링 사용 (osmesa는 macOS/Linux 모두 지원)
# glfw는 메인스레드 제약이 있어 launch_passive에서 문제 발생
if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "glfw"
else:
    os.environ["MUJOCO_GL"] = "egl"

os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

# study 폴더를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# MuJoCo 패치 적용 (환경 빌드 전에 먼저 import 후 패치)
# 이 패치는 quat_rotate 함수 호출 시 w_last=True 인자를 추가합니다
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

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
import json
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig, IsaacRendererWithMuJoco
import torch
from humanoidverse.utils.helpers import export_meta_policy_as_onnx
from humanoidverse.utils.helpers import get_backward_observation
import joblib
import mediapy as media
import numpy as np
from torch.utils._pytree import tree_map
import mujoco
import time
import cv2

import humanoidverse
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent.parent.parent / "humanoidverse"


def main(
    model_folder: Path,
    data_path: Path | None = None,
    headless: bool = False,
    device: str = "cpu",
    simulator: str = "mujoco",
    save_mp4: bool = False,
    disable_dr: bool = False,
    disable_obs_noise: bool = False,
    motion_list: list[int] = [25],
    num_steps: int = 500,
):
    """
    macOS용 tracking inference 스크립트

    Args:
        model_folder: 모델 폴더 경로 (checkpoint/ 포함)
        data_path: 모션 데이터 경로 (선택)
        headless: GUI 없이 실행 (기본값: False, GUI 표시)
        device: cpu (macOS에서는 cuda 미지원)
        simulator: mujoco (macOS에서는 isaacsim 미지원)
        save_mp4: 비디오 저장 여부
        disable_dr: domain randomization 비활성화
        disable_obs_noise: observation noise 비활성화
        motion_list: 평가할 모션 ID 리스트
        num_steps: 시뮬레이션 스텝 수 (headless 모드에서 사용)
    """
    model_folder = Path(model_folder)

    print("Loading model...")
    model = load_model_from_checkpoint_dir(model_folder / "checkpoint", device=device)
    model.to(device)
    model.eval()
    model_name = model.__class__.__name__
    print(f"Model loaded: {model_name}")

    with open(model_folder / "config.json", "r") as f:
        config = json.load(f)

    use_root_height_obs = config["env"].get("root_height_obs", False)

    if data_path is not None:
        config["env"]["lafan_tail_path"] = str(Path(data_path).resolve())
    elif not Path(config["env"].get("lafan_tail_path", "")).exists():
        default_path = HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl"
        if default_path.exists():
            config["env"]["lafan_tail_path"] = str(default_path)
        else:
            config["env"]["lafan_tail_path"] = "data/lafan_29dof.pkl"

    config["env"]["hydra_overrides"].append("env.config.max_episode_length_s=10000")
    config["env"]["hydra_overrides"].append(f"++headless={headless}")
    config["env"]["hydra_overrides"].append(f"simulator={simulator}")
    config["env"]["disable_domain_randomization"] = disable_dr
    config["env"]["disable_obs_noise"] = disable_obs_noise

    # Outputs under model_folder/exported
    output_dir = model_folder / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_meta_policy_as_onnx(
        model,
        output_dir,
        f"{model_name}.onnx",
        {"actor_obs": torch.randn(1, model._actor.input_filter.output_space.shape[0] + model.cfg.archi.z_dim)},
        z_dim=model.cfg.archi.z_dim,
        history=('history_actor' in model.cfg.archi.actor.input_filter.key),
        use_29dof=True,
    )
    print(f"Exported model to {output_dir}/{model_name}.onnx")

    print("Building environment...")
    num_envs = 1
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(num_envs)

    env = wrapped_env._env

    print(f"Environment built successfully")
    print(f"Simulator: {env.config.simulator.config.name}")

    # tracking inference 함수
    def tracking_inference(obs) -> torch.Tensor:
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return model.project_z(z)

    # MuJoCo 모델/데이터 가져오기
    mj_model = env.simulator.model
    mj_data = env.simulator.data

    if not headless:
        print("\n" + "="*60)
        print("Launching MuJoCo viewer with policy control...")
        print("Press 'q' or ESC to exit.")
        print("="*60 + "\n")

        # 모션 설정 및 z 계산
        MOTION_ID = motion_list[0]
        print(f"Motion ID: {MOTION_ID}")
        env.set_is_evaluating(MOTION_ID)

        obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)

        with torch.no_grad():
            z = tracking_inference(obs)
        print(f"Computed z shape: {z.shape}")

        # === 참조 모션 qpos 시퀀스 미리 계산 ===
        # 모션 라이브러리에서 각 프레임의 root_pos, root_rot, dof_pos 추출
        ref_root_pos = obs_dict["ref_body_pos"][:, 0].cpu().numpy()    # [T, 3]
        ref_root_rot = obs_dict["ref_body_rots"][:, 0].cpu().numpy()   # [T, 4] xyzw
        ref_dof_pos = obs_dict["dof_pos"].cpu().numpy()                 # [T, 29]

        # MuJoCo qpos 구성: [root_pos(3), root_quat_wxyz(4), dof_pos(29)] = 36
        # 쿼터니언 변환: xyzw -> wxyz (MuJoCo 포맷)
        ref_root_rot_wxyz = np.roll(ref_root_rot, 1, axis=-1)  # [x,y,z,w] -> [w,x,y,z]
        ref_qpos_seq = np.concatenate([ref_root_pos, ref_root_rot_wxyz, ref_dof_pos], axis=-1)  # [T, 36]
        total_ref_frames = len(ref_qpos_seq)
        print(f"Reference motion: {total_ref_frames} frames, qpos shape: {ref_qpos_seq.shape}")

        # 환경 리셋
        wrapped_obs, _ = wrapped_env.reset(to_numpy=False)

        # GUI 모드: offscreen 렌더링 + cv2 뷰어 (mjpython 불필요)
        RENDER_W, RENDER_H = 640, 480
        mj_model.vis.global_.offwidth = RENDER_W
        mj_model.vis.global_.offheight = RENDER_H

        # 실제 로봇 렌더러
        renderer = mujoco.Renderer(mj_model, height=RENDER_H, width=RENDER_W)

        # 참조 모션 렌더러 (같은 모델의 별도 data 인스턴스)
        ref_mj_data = mujoco.MjData(mj_model)
        ref_renderer = mujoco.Renderer(mj_model, height=RENDER_H, width=RENDER_W)

        # 카메라 설정
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 4.0
        cam.azimuth = 90
        cam.elevation = -20
        cam.lookat[:] = [0, 0, 0.8]

        ref_cam = mujoco.MjvCamera()
        ref_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        ref_cam.distance = 4.0
        ref_cam.azimuth = 90
        ref_cam.elevation = -20
        ref_cam.lookat[:] = [0, 0, 0.8]

        step_counter = 0
        cv2.namedWindow("Tracking Inference (Left: Reference | Right: Policy)", cv2.WINDOW_NORMAL)

        while True:
            # 정책으로 액션 계산
            with torch.no_grad():
                z_current = z[step_counter % len(z)].unsqueeze(0)
                action = model.act(wrapped_obs, z_current, mean=True)

            # 환경 스텝
            wrapped_obs, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)

            # === 참조 모션 렌더링 (왼쪽) ===
            ref_frame_idx = step_counter % total_ref_frames
            ref_qpos = ref_qpos_seq[ref_frame_idx]
            ref_mj_data.qpos[:] = ref_qpos
            ref_mj_data.qvel[:] = 0  # 속도는 0으로 설정
            mujoco.mj_forward(mj_model, ref_mj_data)  # FK 계산

            ref_cam.lookat[:] = ref_mj_data.qpos[:3]
            ref_cam.lookat[2] = 0.8
            ref_renderer.update_scene(ref_mj_data, camera=ref_cam)
            ref_frame = ref_renderer.render()

            # === 실제 로봇 렌더링 (오른쪽) ===
            cam.lookat[:] = mj_data.qpos[:3]
            cam.lookat[2] = 0.8
            renderer.update_scene(mj_data, camera=cam)
            policy_frame = renderer.render()

            # 텍스트 오버레이
            ref_frame_bgr = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2BGR)
            policy_frame_bgr = cv2.cvtColor(policy_frame, cv2.COLOR_RGB2BGR)

            cv2.putText(ref_frame_bgr, "Reference Motion", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(ref_frame_bgr, f"Frame {ref_frame_idx}/{total_ref_frames}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

            cv2.putText(policy_frame_bgr, "Policy Output", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(policy_frame_bgr, f"Step {step_counter}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # 좌우로 합치기
            combined = np.hstack([ref_frame_bgr, policy_frame_bgr])
            cv2.imshow("Tracking Inference (Left: Reference | Right: Policy)", combined)

            step_counter += 1

            # 에피소드 종료 시 리셋
            if terminated or truncated:
                wrapped_obs, _ = wrapped_env.reset(to_numpy=False)
                step_counter = 0

            # 키 입력 확인 (50Hz, 20ms 대기)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 또는 ESC
                break

        cv2.destroyAllWindows()
        renderer.close()
        ref_renderer.close()
        print(f"\nViewer closed after {step_counter} steps")

    else:
        # Headless 모드: 추론 실행
        print(f"\nRunning inference for {num_steps} steps...")

        for MOTION_ID in motion_list:
            print(f"\nMotion ID: {MOTION_ID}")
            env.set_is_evaluating(MOTION_ID)

            obs, obs_dict = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)

            # z 계산
            with torch.no_grad():
                z = tracking_inference(obs)

            print(f"  Computed z shape: {z.shape}")

            # 롤아웃
            wrapped_obs, _ = wrapped_env.reset()
            frames = []

            for step in range(num_steps):
                # 액션 계산
                obs_tensor = {k: torch.tensor(v, device=device, dtype=torch.float32)
                              for k, v in wrapped_obs.items() if k != 'history'}

                with torch.no_grad():
                    action = model.act(obs_tensor, z=z[:1], mean=True)

                # 환경 스텝
                wrapped_obs, reward, terminated, truncated, info = wrapped_env.step(action.cpu().numpy())

                if step % 100 == 0:
                    print(f"  Step {step}/{num_steps}")

            print(f"  Rollout complete")

    print("\nDone!")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
