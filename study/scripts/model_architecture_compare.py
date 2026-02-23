"""
실습 8: 모델 아키텍처 비교

다양한 hidden_dim/hidden_layers 설정에서 파라미터 수와 추론 속도를 비교합니다.

사용법:
    uv run python study/scripts/model_architecture_compare.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import gymnasium

from humanoidverse.agents.fb_cpr.model import FBcprModel, FBcprModelConfig
from humanoidverse.agents.nn_models import (
    ForwardArchiConfig,
    ActorArchiConfig,
    BackwardArchiConfig,
)


def count_params(model):
    """모델의 전체 파라미터 수를 반환합니다."""
    return sum(p.numel() for p in model.parameters())


def count_module_params(model):
    """모듈별 파라미터 수를 반환합니다."""
    return {
        "Forward Map": sum(p.numel() for p in model._forward_map.parameters()),
        "Backward Map": sum(p.numel() for p in model._backward_map.parameters()),
        "Actor": sum(p.numel() for p in model._actor.parameters()),
        "Critic": sum(p.numel() for p in model._critic.parameters()),
        "Discriminator": sum(p.numel() for p in model._discriminator.parameters()),
    }


def benchmark_inference(model, obs, z, n_iters=100):
    """추론 시간을 벤치마크합니다."""
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model.act(obs, z, mean=True)

    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            model.act(obs, z, mean=True)
    elapsed = time.perf_counter() - start

    return elapsed / n_iters * 1000  # ms per inference


def main():
    # 관측 공간 정의
    obs_space = gymnasium.spaces.Dict({
        "state": gymnasium.spaces.Box(low=-10, high=10, shape=(64,)),
        "privileged_state": gymnasium.spaces.Box(low=-10, high=10, shape=(357,)),
        "last_action": gymnasium.spaces.Box(low=-1, high=1, shape=(29,)),
    })
    action_dim = 29
    batch = 16

    # 테스트할 아키텍처 설정들
    configs = [
        {"name": "Small", "hidden_dim": 512, "hidden_layers": 2, "z_dim": 256},
        {"name": "Medium", "hidden_dim": 1024, "hidden_layers": 2, "z_dim": 256},
        {"name": "Large (default)", "hidden_dim": 2048, "hidden_layers": 6, "z_dim": 256},
        {"name": "Small z_dim=128", "hidden_dim": 1024, "hidden_layers": 2, "z_dim": 128},
        {"name": "Large z_dim=512", "hidden_dim": 2048, "hidden_layers": 6, "z_dim": 512},
    ]

    print("=" * 80)
    print("BFM-Zero 모델 아키텍처 비교")
    print("=" * 80)
    print()

    results = []

    for cfg_dict in configs:
        name = cfg_dict["name"]
        hd = cfg_dict["hidden_dim"]
        hl = cfg_dict["hidden_layers"]
        zd = cfg_dict["z_dim"]

        print(f"Building '{name}' (hidden_dim={hd}, hidden_layers={hl}, z_dim={zd})...")

        try:
            model_cfg = FBcprModelConfig(
                device="cpu",
                archi=FBcprModelConfig.__fields__["archi"].default.__class__(
                    z_dim=zd,
                    f=ForwardArchiConfig(hidden_dim=hd, hidden_layers=hl),
                    actor=ActorArchiConfig(hidden_dim=hd, hidden_layers=hl),
                    b=BackwardArchiConfig(hidden_dim=hd, hidden_layers=hl),
                ),
            )
            model = model_cfg.build(obs_space, action_dim)

            total = count_params(model)
            modules = count_module_params(model)

            # 더미 데이터로 추론
            obs = {k: torch.randn(batch, v.shape[0]) for k, v in obs_space.spaces.items()}
            z = model.sample_z(batch)

            # 추론 벤치마크
            ms_per_inf = benchmark_inference(model, obs, z)

            results.append({
                "name": name,
                "total_params": total,
                "modules": modules,
                "ms_per_inference": ms_per_inf,
                "z_dim": zd,
            })

            print(f"  Total params: {total:,}")
            print(f"  Inference: {ms_per_inf:.2f} ms/batch")
            print()

        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    # 결과 테이블
    if results:
        print("=" * 80)
        print(f"{'Name':<25} {'Params':>12} {'Inference':>12} {'z_dim':>8}")
        print(f"{'─' * 25} {'─' * 12} {'─' * 12} {'─' * 8}")

        for r in results:
            params_str = f"{r['total_params'] / 1e6:.1f}M"
            inf_str = f"{r['ms_per_inference']:.2f} ms"
            print(f"{r['name']:<25} {params_str:>12} {inf_str:>12} {r['z_dim']:>8}")

        print()
        print("모듈별 파라미터 분포 (Large default):")
        for r in results:
            if "default" in r["name"]:
                total = r["total_params"]
                for mod_name, mod_params in r["modules"].items():
                    pct = mod_params / total * 100
                    print(f"  {mod_name:<20} {mod_params:>12,} ({pct:.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
