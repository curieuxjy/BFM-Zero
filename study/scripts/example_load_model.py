"""
모델 로딩 예제 스크립트

사용법:
    uv run python study/scripts/example_load_model.py --model_folder model/
"""

import sys
from pathlib import Path

# humanoidverse 모듈 임포트를 위해 상위 디렉토리 추가
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json
import torch
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir


def main(model_folder: str = "model", device: str = "cuda"):
    model_folder = Path(model_folder)

    # 설정 파일 로드
    config_path = model_folder / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print("=== Config ===")
        print(f"Agent: {config.get('agent', {}).get('name', 'Unknown')}")
        print(f"Env: {config.get('env', {}).get('name', 'Unknown')}")

    # 모델 로드
    checkpoint_dir = model_folder / "checkpoint"
    if checkpoint_dir.exists():
        print(f"\n=== Loading model from {checkpoint_dir} ===")
        model = load_model_from_checkpoint_dir(checkpoint_dir, device=device)
        model.eval()

        print(f"Model class: {model.__class__.__name__}")
        print(f"Device: {next(model.parameters()).device}")

        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    else:
        print(f"Checkpoint directory not found: {checkpoint_dir}")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
