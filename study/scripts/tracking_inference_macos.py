"""
macOS용 tracking inference 스크립트

사용법:
    uv run python study/scripts/tracking_inference_macos.py --model_folder model/
    uv run python study/scripts/tracking_inference_macos.py --model_folder model/ --no-headless

이 스크립트는 메인 소스코드를 수정하지 않고 런타임 패치를 적용합니다.
"""
import os
import sys

# macOS에서는 glfw 사용 (egl은 Linux 전용)
if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "glfw"
else:
    os.environ["MUJOCO_GL"] = "egl"

os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

# study 폴더를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# 패치 적용 (humanoidverse import 전에)
from study.patches.config_patch import apply_config_patches
apply_config_patches()

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
import mujoco.viewer
import time

import humanoidverse
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent.parent.parent / "humanoidverse"


def main(
    model_folder: Path,
    data_path: Path | None = None,
    headless: bool = True,
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
        headless: GUI 없이 실행 (--no-headless로 GUI 표시)
        device: cpu (macOS에서는 cuda 미지원)
        simulator: mujoco (macOS에서는 isaacsim 미지원)
        save_mp4: 비디오 저장 여부
        disable_dr: domain randomization 비활성화
        disable_obs_noise: observation noise 비활성화
        motion_list: 평가할 모션 ID 리스트
        num_steps: 시뮬레이션 스텝 수
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

    # MuJoCo 패치 적용 (환경 빌드 전에 - MuJoCo 클래스가 로드된 후 인스턴스화 전에)
    # HumanoidVerseIsaacConfig import 시 MuJoCo 모듈이 로드되므로 이 시점에 패치 가능
    from study.patches.mujoco_patch import apply_mujoco_patches
    apply_mujoco_patches()

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
        print("Launching MuJoCo viewer...")
        print("Close the window to exit.")
        print("="*60 + "\n")

        # 환경 리셋
        obs, _ = wrapped_env.reset()

        # GUI 모드: mujoco.viewer.launch 사용
        def controller(model, data):
            """뷰어에서 호출되는 컨트롤러"""
            pass  # 시뮬레이션은 환경에서 처리

        mujoco.viewer.launch(mj_model, mj_data)

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
