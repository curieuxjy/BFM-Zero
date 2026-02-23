"""
실습 6: 마찰 계수 실험

MuJoCo의 geom_friction 값을 변경하면서 트래킹 성능을 비교합니다.

사용법:
    uv run python study/scripts/friction_experiment.py --model_folder model/
    uv run python study/scripts/friction_experiment.py --model_folder model/ --friction_scales 0.3 0.5 1.0 1.5 2.0
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


def run_rollout(wrapped_env, env, model, z, num_steps: int, friction_scale: float):
    """주어진 마찰 계수로 롤아웃을 실행하고 통계를 반환합니다."""
    mj_model = env.simulator.model
    mj_data = env.simulator.data

    # 마찰 계수 변경
    original_friction = mj_model.geom_friction.copy()
    mj_model.geom_friction[:] = original_friction * friction_scale

    wrapped_obs, _ = wrapped_env.reset(to_numpy=False)

    positions = []
    heights = []
    alive_steps = 0

    for step in range(num_steps):
        with torch.no_grad():
            z_current = z[step % len(z)].unsqueeze(0)
            action = model.act(wrapped_obs, z_current, mean=True)

        wrapped_obs, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)

        pos = mj_data.qpos[:3].copy()
        positions.append(pos[:2])
        heights.append(pos[2])
        alive_steps += 1

        if terminated or truncated:
            break

    # 원래 마찰 복원
    mj_model.geom_friction[:] = original_friction

    positions = np.array(positions)
    heights = np.array(heights)

    return {
        "friction_scale": friction_scale,
        "alive_steps": alive_steps,
        "mean_height": float(heights.mean()),
        "min_height": float(heights.min()),
        "total_distance": float(np.linalg.norm(positions[-1] - positions[0])) if len(positions) > 1 else 0,
        "fell": alive_steps < num_steps,
    }


def main(
    model_folder: Path,
    motion_id: int = 25,
    friction_scales: list[float] = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    num_steps: int = 300,
    device: str = "cpu",
):
    """
    마찰 계수에 따른 트래킹 성능을 비교합니다.

    Args:
        model_folder: 모델 폴더 경로
        motion_id: 트래킹할 모션 ID
        friction_scales: 테스트할 마찰 스케일 목록
        num_steps: 롤아웃 스텝 수
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
    config["env"]["disable_domain_randomization"] = True
    config["env"]["disable_obs_noise"] = True

    # 환경 빌드
    print("Building environment...")
    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, _ = env_cfg.build(1)
    env = wrapped_env._env

    # z 계산
    print(f"Computing z for motion {motion_id}...")
    env.set_is_evaluating(motion_id)
    obs, _ = get_backward_observation(env, 0, use_root_height_obs=use_root_height_obs)

    with torch.no_grad():
        z = model.backward_map(obs)
        for step in range(z.shape[0]):
            end_idx = min(step + 1, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        z = model.project_z(z)

    # 마찰 실험
    print(f"\n{'=' * 60}")
    print(f"마찰 계수 실험 (Motion {motion_id}, {num_steps} steps)")
    print(f"{'=' * 60}")

    results = []
    for scale in friction_scales:
        print(f"\n  Friction scale = {scale:.1f}...")
        result = run_rollout(wrapped_env, env, model, z, num_steps, scale)
        results.append(result)

    # 결과 테이블
    print(f"\n{'=' * 70}")
    print(f"{'Friction':>10} {'Steps':>8} {'Height':>10} {'Min H':>10} {'Distance':>10} {'Status':>10}")
    print(f"{'─' * 10} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")

    for r in results:
        status = "FELL" if r["fell"] else "OK"
        print(f"{r['friction_scale']:>10.1f} "
              f"{r['alive_steps']:>8d} "
              f"{r['mean_height']:>10.3f} "
              f"{r['min_height']:>10.3f} "
              f"{r['total_distance']:>10.3f} "
              f"{status:>10}")

    print(f"\n예상: 0.8~1.2 범위에서 가장 안정적, 극단값(0.3, 2.0)에서 성능 저하")
    print("Done!")


if __name__ == "__main__":
    tyro.cli(main)
