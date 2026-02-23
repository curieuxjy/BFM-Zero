"""
실습 3: z 벡터 분석

여러 모션 ID에 대한 z 벡터를 추출하고 코사인 유사도를 비교합니다.

사용법:
    uv run python study/scripts/analyze_z_vectors.py --model_folder model/
    uv run python study/scripts/analyze_z_vectors.py --model_folder model/ --motion_ids 0 5 10 25 50
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

import json
import numpy as np
import tyro

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig
from humanoidverse.utils.helpers import get_backward_observation

import humanoidverse
if getattr(humanoidverse, "__file__", None) is not None:
    HUMANOIDVERSE_DIR = Path(humanoidverse.__file__).parent
else:
    HUMANOIDVERSE_DIR = Path(__file__).resolve().parent.parent.parent / "humanoidverse"


def main(
    model_folder: Path,
    motion_ids: list[int] = [0, 5, 10, 25, 50],
    device: str = "cpu",
):
    """
    여러 모션의 z 벡터를 추출하고 코사인 유사도를 비교합니다.

    Args:
        model_folder: 모델 폴더 경로
        motion_ids: 분석할 모션 ID 리스트
        device: cpu
    """
    model_folder = Path(model_folder)

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
    config["env"]["hydra_overrides"].append("++headless=True")
    config["env"]["hydra_overrides"].append("simulator=mujoco")

    # 환경 빌드
    print("Building environment...")
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(1)
    env = wrapped_env._env

    # 각 모션의 z 벡터 추출
    z_dict = {}
    for mid in motion_ids:
        try:
            env.set_is_evaluating(mid)
            obs, _ = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)

            with torch.no_grad():
                z = model.backward_map(obs)
                z = model.project_z(z)

            z_np = z.cpu().numpy()
            z_dict[mid] = z_np
            print(f"  Motion {mid}: z shape={z_np.shape}, ||z||={np.linalg.norm(z_np, axis=-1).mean():.2f}")
        except Exception as e:
            print(f"  Motion {mid}: FAILED - {e}")

    # 코사인 유사도 분석
    print("\n" + "=" * 60)
    print("=== 코사인 유사도 행렬 (첫 프레임 z 기준) ===")
    print("=" * 60)

    keys = list(z_dict.keys())
    # 헤더
    header = f"{'':>10}"
    for k in keys:
        header += f"{'M'+str(k):>10}"
    print(header)

    for i, ki in enumerate(keys):
        row = f"{'M'+str(ki):>10}"
        for j, kj in enumerate(keys):
            zi = z_dict[ki][0]
            zj = z_dict[kj][0]
            cos_sim = np.dot(zi, zj) / (np.linalg.norm(zi) * np.linalg.norm(zj) + 1e-8)
            row += f"{cos_sim:>10.4f}"
        print(row)

    # 시간에 따른 z 변화 분석
    print("\n" + "=" * 60)
    print("=== 시간에 따른 z 변화 (프레임 간 코사인 유사도) ===")
    print("=" * 60)

    for mid in keys:
        z_seq = z_dict[mid]
        if len(z_seq) < 2:
            continue
        consecutive_sims = []
        for t in range(len(z_seq) - 1):
            cos = np.dot(z_seq[t], z_seq[t + 1]) / (
                np.linalg.norm(z_seq[t]) * np.linalg.norm(z_seq[t + 1]) + 1e-8
            )
            consecutive_sims.append(cos)
        sims = np.array(consecutive_sims)
        print(f"  Motion {mid}: mean={sims.mean():.4f}, min={sims.min():.4f}, max={sims.max():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    tyro.cli(main)
