"""
실습 1: 텐서 Shape 추적

CPU 더미 데이터로 FBcprModel의 주요 텐서 shape을 확인합니다.

사용법:
    uv run python study/scripts/debug_tensor_shapes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import gymnasium

from humanoidverse.agents.fb_cpr.model import FBcprModel, FBcprModelConfig


def main():
    # 관측 공간 정의 (G1 29-DOF 기본 설정)
    obs_space = gymnasium.spaces.Dict({
        "state": gymnasium.spaces.Box(low=-10, high=10, shape=(64,)),
        "privileged_state": gymnasium.spaces.Box(low=-10, high=10, shape=(357,)),
        "last_action": gymnasium.spaces.Box(low=-1, high=1, shape=(29,)),
    })
    action_dim = 29

    # 모델 생성
    cfg = FBcprModelConfig(device="cpu")
    model = cfg.build(obs_space, action_dim)
    model._prepare_for_train()

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # 더미 배치 데이터
    batch = 16
    obs = {k: torch.randn(batch, v.shape[0]) for k, v in obs_space.spaces.items()}
    z = model.sample_z(batch)
    action = torch.randn(batch, action_dim)

    print("=" * 60)
    print("=== 입력 텐서 Shape ===")
    print("=" * 60)
    for k, v in obs.items():
        print(f"  obs[{k}]: {v.shape}")
    print(f"  z: {z.shape}")
    print(f"  ||z||: {z.norm(dim=-1).mean():.2f} (expected: {z.shape[-1]**0.5:.2f})")
    print(f"  action: {action.shape}")
    print()

    print("=" * 60)
    print("=== 네트워크 출력 Shape ===")
    print("=" * 60)

    with torch.no_grad():
        # Backward Map: obs -> z_dim
        B = model._backward_map(obs)
        print(f"  B(obs) [Backward Map]: {B.shape}")

        # Forward Map: (obs, z, action) -> (num_parallel, batch, z_dim)
        F_out = model._forward_map(obs, z, action)
        print(f"  F(obs, z, a) [Forward Map]: {F_out.shape}")

        # Successor Measure: M = F @ B^T
        M = torch.matmul(F_out, B.T)
        print(f"  M(s,g) = F @ B^T: {M.shape}")

        # Actor: (obs, z) -> action
        act = model.act(obs, z, mean=True)
        print(f"  Actor(obs, z): {act.shape}")

        # Critic: (obs, z, action) -> Q-value
        Q = model._critic(obs, z, action)
        print(f"  Critic(obs, z, a): {Q.shape}")

        # Discriminator: (obs, z) -> logits
        D = model._discriminator.compute_logits(obs, z)
        print(f"  Discriminator(obs, z): {D.shape}")

    print()
    print("=" * 60)
    print("=== 텐서 Shape 요약 ===")
    print("=" * 60)
    print(f"  {'텐서':<30} {'Shape':<20} {'설명'}")
    print(f"  {'─'*30} {'─'*20} {'─'*30}")
    print(f"  {'obs[state]':<30} {str(obs['state'].shape):<20} dof_pos+dof_vel+gravity+ang_vel")
    print(f"  {'obs[privileged_state]':<30} {str(obs['privileged_state'].shape):<20} 전체 강체 관측")
    print(f"  {'obs[last_action]':<30} {str(obs['last_action'].shape):<20} 이전 액션")
    print(f"  {'z':<30} {str(z.shape):<20} 행동 의도 벡터")
    print(f"  {'F(s,a)':<30} {str(F_out.shape):<20} 앙상블 Forward 출력")
    print(f"  {'B(g)':<30} {str(B.shape):<20} Backward 출력")
    print(f"  {'M(s,g)':<30} {str(M.shape):<20} Successor Measure 행렬")
    print(f"  {'Q_critic':<30} {str(Q.shape):<20} 앙상블 Q-value")
    print(f"  {'D_logits':<30} {str(D.shape):<20} Discriminator 로짓")
    print(f"  {'action':<30} {str(act.shape):<20} 정책 출력 액션")


if __name__ == "__main__":
    main()
