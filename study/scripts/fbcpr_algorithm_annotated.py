"""
FBcpr 알고리즘 상세 주석 코드

이 파일은 FBcpr (Forward-Backward with Contrastive Predictive Representation) 알고리즘의
핵심 로직을 상세하게 주석 처리한 학습/분석용 코드입니다.

원본 파일들:
- humanoidverse/agents/fb/agent.py
- humanoidverse/agents/fb/model.py
- humanoidverse/agents/fb_cpr/agent.py
- humanoidverse/agents/fb_cpr/model.py
- humanoidverse/agents/fb_cpr_aux/agent.py
- humanoidverse/agents/fb_cpr_aux/model.py
- humanoidverse/agents/nn_models.py

=============================================================================
FBcpr 알고리즘 개요
=============================================================================

FB (Forward-Backward) 표현 학습:
- Successor Feature를 학습하여 상태-목표 관계를 잠재 공간에서 표현
- Forward Map: 미래 상태 예측
- Backward Map: 현재 상태를 잠재 벡터로 인코딩

CPR (Contrastive Predictive Representation):
- 판별자(Discriminator)로 전문가 vs 정책 구분
- GAN 스타일 학습으로 암묵적 보상 생성

핵심 아이디어:
1. z = B(goal): 목표 상태를 잠재 벡터로 인코딩
2. F(s, z, a): (상태, 목표, 액션)에서 미래 상태 특징 예측
3. M = F · B^T: Successor Feature Matrix (상태-목표 연결성)
4. π(s, z): z 조건부 정책 학습
5. D(s, z): 전문가/정책 판별 → 보상 신호

수학적 배경:
- Successor Feature: ψ(s, a) = E[Σ γ^t φ(s_t) | s_0=s, a_0=a]
- 보상: r(s) = φ(s)^T w (특징 가중치)
- FB 학습: M(s, g) = E[γ^T | s_0=s, goal=g] ≈ F(s) · B(g)^T

=============================================================================
"""

import copy
import math
import typing as tp
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.amp import autocast


# =============================================================================
# 1. 기본 유틸리티
# =============================================================================

def _soft_update_params(net_params: tp.Any, target_net_params: tp.Any, tau: float):
    """
    Target 네트워크 Soft Update

    θ_target = (1 - τ) * θ_target + τ * θ

    Args:
        net_params: 현재 네트워크 파라미터
        target_net_params: 타겟 네트워크 파라미터
        tau: 업데이트 비율 (0.01 ~ 0.1)

    이유:
    - 학습 안정성을 위해 타겟 네트워크를 천천히 업데이트
    - TD Learning에서 목표값이 급격히 변하면 발산 가능
    """
    torch._foreach_mul_(target_net_params, 1 - tau)  # θ_target *= (1 - τ)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)  # θ_target += τ * θ


class eval_mode:
    """평가 모드 컨텍스트 매니저"""
    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


# =============================================================================
# 2. 신경망 구성요소
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1 Backward Map: obs → z
# -----------------------------------------------------------------------------

class BackwardMap(nn.Module):
    """
    Backward Map: 관측값을 잠재 벡터 z로 인코딩

    역할:
    - 목표 상태 인코딩: z = B(goal)
    - 전문가 궤적 인코딩: z = B(expert_obs)

    아키텍처:
    - Input: obs (state + privileged_state)
    - Hidden: 256-dim, 1 layer
    - Output: z (256-dim)
    - Normalization: L2 norm (optional)

    수학:
    - B: S → Z
    - z = B(s) = normalize(MLP(s))
    """
    def __init__(self, obs_dim: int, z_dim: int, hidden_dim: int = 256, hidden_layers: int = 1, norm: bool = True):
        super().__init__()

        # 첫 번째 레이어: Linear + LayerNorm + Tanh
        layers = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]

        # 중간 레이어: Linear + ReLU
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]

        # 출력 레이어
        layers += [nn.Linear(hidden_dim, z_dim)]

        # L2 정규화 (선택)
        if norm:
            layers += [Norm()]

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: 관측값 [batch, obs_dim]

        Returns:
            z: 잠재 벡터 [batch, z_dim]
        """
        return self.net(obs)


class Norm(nn.Module):
    """L2 정규화 레이어"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)


# -----------------------------------------------------------------------------
# 2.2 Forward Map: (obs, z, action) → successor feature
# -----------------------------------------------------------------------------

class ForwardMap(nn.Module):
    """
    Forward Map: 상태-액션-목표에서 미래 상태 특징 예측

    역할:
    - Successor Feature 학습: F(s, z, a) ≈ E[γ^T B(s_T) | s, z, a]
    - Q-value 계산: Q(s, z, a) = F(s, z, a) · z

    아키텍처:
    - Input: (obs, z) 임베딩 + (obs, action) 임베딩
    - Hidden: 2048-dim, 6 layers (Residual)
    - Output: z_dim (256)
    - Ensemble: num_parallel=2 (불확실성 추정)

    수학:
    - F: S × Z × A → Z
    - Q(s, z, a) = F(s, z, a)^T · z
    - Target: γ F(s', z, a') + B(s) · z
    """
    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        action_dim: int,
        hidden_dim: int = 2048,
        hidden_layers: int = 6,
        embedding_layers: int = 2,
        num_parallel: int = 2,  # 앙상블 (불확실성)
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_parallel = num_parallel
        self.hidden_dim = hidden_dim

        # (obs, z) 임베딩 네트워크
        # 목표 조건부 상태 표현 학습
        self.embed_z = self._build_embedding(obs_dim + z_dim, hidden_dim, embedding_layers, num_parallel)

        # (obs, action) 임베딩 네트워크
        # 상태-액션 표현 학습
        self.embed_sa = self._build_embedding(obs_dim + action_dim, hidden_dim, embedding_layers, num_parallel)

        # 메인 네트워크: 두 임베딩을 결합하여 출력
        layers = []
        for _ in range(hidden_layers):
            layers += [self._parallel_linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
        layers += [self._parallel_linear(hidden_dim, z_dim, num_parallel)]
        self.Fs = nn.Sequential(*layers)

    def _build_embedding(self, input_dim, hidden_dim, layers, num_parallel):
        """임베딩 네트워크 생성"""
        seq = [self._parallel_linear(input_dim, hidden_dim, num_parallel),
               self._parallel_layernorm(hidden_dim, num_parallel), nn.Tanh()]
        for _ in range(layers - 2):
            seq += [self._parallel_linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
        seq += [self._parallel_linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]
        return nn.Sequential(*seq)

    def _parallel_linear(self, in_dim, out_dim, num_parallel):
        """앙상블용 병렬 Linear 레이어"""
        if num_parallel > 1:
            return DenseParallel(in_dim, out_dim, num_parallel)
        return nn.Linear(in_dim, out_dim)

    def _parallel_layernorm(self, dim, num_parallel):
        """앙상블용 병렬 LayerNorm"""
        if num_parallel > 1:
            return ParallelLayerNorm([dim], num_parallel)
        return nn.LayerNorm(dim)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: 관측값 [batch, obs_dim]
            z: 잠재 벡터 [batch, z_dim]
            action: 액션 [batch, action_dim]

        Returns:
            F: Successor feature [num_parallel, batch, z_dim]
        """
        # 앙상블을 위해 차원 확장
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)

        # (obs, z) 임베딩: 목표 조건부 상태 표현
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # [num_parallel, batch, hidden/2]

        # (obs, action) 임베딩: 상태-액션 표현
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))  # [num_parallel, batch, hidden/2]

        # 결합하여 최종 출력
        combined = torch.cat([sa_embedding, z_embedding], dim=-1)  # [num_parallel, batch, hidden]
        return self.Fs(combined)  # [num_parallel, batch, z_dim]


# -----------------------------------------------------------------------------
# 2.3 Actor: (obs, z) → action
# -----------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Actor: z 조건부 정책

    역할:
    - 주어진 z에 따라 행동 결정
    - z가 다르면 다른 행동 (목표/태스크 조건부)

    아키텍처:
    - Input: (obs, z) 임베딩 + obs 임베딩
    - Hidden: 2048-dim, 6 layers (Residual)
    - Output: action (29-dim for G1)
    - Distribution: TruncatedNormal (std=0.05)

    수학:
    - π: S × Z → Δ(A)
    - a ~ π(·|s, z) = N(μ(s, z), σ²)
    """
    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        action_dim: int,
        hidden_dim: int = 2048,
        hidden_layers: int = 6,
        embedding_layers: int = 2,
    ):
        super().__init__()

        # (obs, z) 임베딩: 목표 조건부 상태 표현
        self.embed_z = self._build_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)

        # obs 임베딩: 순수 상태 표현
        self.embed_s = self._build_embedding(obs_dim, hidden_dim, embedding_layers)

        # 정책 네트워크
        layers = []
        for _ in range(hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, action_dim)]
        self.policy = nn.Sequential(*layers)

    def _build_embedding(self, input_dim, hidden_dim, layers):
        """임베딩 네트워크 생성"""
        seq = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(layers - 2):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU()]
        return nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, std: float) -> "TruncatedNormal":
        """
        Args:
            obs: 관측값 [batch, obs_dim]
            z: 잠재 벡터 [batch, z_dim]
            std: 표준편차 (탐색 노이즈)

        Returns:
            dist: 액션 분포 TruncatedNormal(μ, σ)
        """
        # (obs, z) 임베딩
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # [batch, hidden/2]

        # obs 임베딩
        s_embedding = self.embed_s(obs)  # [batch, hidden/2]

        # 결합
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)  # [batch, hidden]

        # 정책 출력 (tanh로 [-1, 1] 범위)
        mu = torch.tanh(self.policy(embedding))

        # 분포 생성
        std = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std)


# -----------------------------------------------------------------------------
# 2.4 Discriminator: (obs, z) → expert_prob
# -----------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Discriminator: 전문가 vs 정책 판별

    역할:
    - GAN 스타일 학습으로 암묵적 보상 생성
    - D(s, z) → P(expert | s, z)
    - 보상: r = log D - log(1-D)

    아키텍처:
    - Input: (obs, z) concat
    - Hidden: 1024-dim, 3 layers
    - Output: logit (1-dim)

    수학:
    - D: S × Z → [0, 1]
    - r(s, z) = log D(s, z) - log(1 - D(s, z))

    이유:
    - 전문가 데이터와 정책 데이터를 구분하는 암묵적 보상
    - GAIL (Generative Adversarial Imitation Learning) 스타일
    """
    def __init__(self, obs_dim: int, z_dim: int, hidden_dim: int = 1024, hidden_layers: int = 3):
        super().__init__()

        # 네트워크: Linear + LayerNorm + Tanh → Linear + ReLU → ... → Linear
        layers = [nn.Linear(obs_dim + z_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1)]  # 출력: logit

        self.trunk = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """확률 출력"""
        logits = self.compute_logits(obs, z)
        return torch.sigmoid(logits)

    def compute_logits(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Logit 출력"""
        x = torch.cat([z, obs], dim=1)
        return self.trunk(x)

    def compute_reward(self, obs: torch.Tensor, z: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        암묵적 보상 계산

        r = log D(s, z) - log(1 - D(s, z))
          = log(P(expert) / P(policy))

        직관:
        - 전문가처럼 보이면 높은 보상
        - 정책처럼 보이면 낮은 보상
        """
        D = self.forward(obs, z)
        D = torch.clamp(D, eps, 1 - eps)  # 수치 안정성
        reward = D.log() - (1 - D).log()  # log odds ratio
        return reward


# =============================================================================
# 3. FBModel - 기본 FB 모델
# =============================================================================

class FBModel(nn.Module):
    """
    Forward-Backward 모델

    구성요소:
    - backward_map: obs → z
    - forward_map: (obs, z, action) → successor feature
    - actor: (obs, z) → action
    - obs_normalizer: 관측값 정규화

    Target Networks:
    - target_backward_map: soft update
    - target_forward_map: soft update
    """
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, device: str = "cuda"):
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        self.norm_z = True  # z 정규화 여부

        # 핵심 네트워크
        self._backward_map = BackwardMap(obs_dim, z_dim)
        self._forward_map = ForwardMap(obs_dim, z_dim, action_dim)
        self._actor = Actor(obs_dim, z_dim, action_dim)

        # 관측값 정규화 (BatchNorm)
        self._obs_normalizer = nn.Identity()  # 실제로는 BatchNorm

        self.to(device)

    def _prepare_for_train(self) -> None:
        """Target 네트워크 생성"""
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        """
        균등 분포에서 z 샘플링

        z ~ N(0, I) → normalize → sqrt(z_dim) * z

        이유:
        - 단위 구 위에서 균등하게 분포
        - ||z|| = sqrt(z_dim) 유지
        """
        z = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        z를 단위 구에 투영

        z → sqrt(z_dim) * z / ||z||

        이유:
        - z의 norm을 일정하게 유지
        - 방향만 의미 있는 정보
        """
        if self.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor) -> torch.Tensor:
        """obs → z"""
        return self._backward_map(self._obs_normalizer(obs))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(obs, z, action) → successor feature"""
        return self._forward_map(self._obs_normalizer(obs), z, action)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True, std: float = 0.05) -> torch.Tensor:
        """
        액션 선택

        Args:
            obs: 관측값
            z: 잠재 벡터 (목표/태스크 인코딩)
            mean: True면 평균 액션, False면 샘플링
            std: 탐색 노이즈 표준편차
        """
        dist = self._actor(self._obs_normalizer(obs), z, std)
        if mean:
            return dist.mean.float()
        return dist.sample().float()

    # -------------------------------------------------------------------------
    # 추론 메서드
    # -------------------------------------------------------------------------

    def tracking_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        """
        모션 트래킹용 z 추론

        주어진 궤적의 각 프레임에서 z를 계산하고,
        미래 seq_length 프레임의 평균을 사용

        Args:
            next_obs: 전문가 궤적 [T, obs_dim]

        Returns:
            z: 각 프레임의 z [T, z_dim]

        알고리즘:
        1. 각 프레임에 대해 z = B(obs) 계산
        2. 각 z는 미래 seq_length 프레임의 평균으로 교체
           z[t] = mean(z[t:t+seq_length])
        """
        z = self.backward_map(next_obs)  # [T, z_dim]

        seq_length = 8  # 실제 설정값
        for step in range(z.shape[0]):
            end_idx = min(step + seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)

        return self.project_z(z)

    def goal_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        """
        목표 도달용 z 추론

        목표 프레임에서 직접 z 계산

        Args:
            next_obs: 목표 상태 [1, obs_dim]

        Returns:
            z: 목표 z [1, z_dim]
        """
        z = self.backward_map(next_obs)
        return self.project_z(z)

    def reward_inference(
        self,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        보상 기반 z 추론

        보상 가중 평균으로 z 계산
        z = Σ r_i * B(s_i) / Σ r_i

        Args:
            next_obs: 샘플 상태들 [N, obs_dim]
            reward: 각 상태의 보상 [N, 1]
            weight: 가중치 (softmax 등)

        Returns:
            z: 보상 최적화 z [1, z_dim]

        이유:
        - 높은 보상을 받은 상태의 z를 더 많이 반영
        - 보상 함수를 z 공간에서 표현
        """
        z = 0
        wr = reward if weight is None else reward * weight

        B = self.backward_map(next_obs)  # [N, z_dim]
        z = torch.matmul(wr.T, B)  # [1, z_dim]

        return self.project_z(z)


# =============================================================================
# 4. FBcprModel - Discriminator 추가
# =============================================================================

class FBcprModel(FBModel):
    """
    FBcpr 모델 (FB + Discriminator + Critic)

    추가 구성요소:
    - discriminator: 전문가/정책 판별
    - critic: Q-function (판별자 보상용)
    """
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, device: str = "cuda"):
        super().__init__(obs_dim, z_dim, action_dim, device)

        # 판별자: 암묵적 보상 생성
        self._discriminator = Discriminator(obs_dim, z_dim)

        # Critic: 판별자 보상의 Q-value
        self._critic = ForwardMap(obs_dim, z_dim, action_dim, output_dim=1)

        self.to(device)

    def _prepare_for_train(self) -> None:
        """Target 네트워크 생성"""
        super()._prepare_for_train()
        self._target_critic = copy.deepcopy(self._critic)


# =============================================================================
# 5. FBcprAuxModel - Auxiliary Rewards 추가
# =============================================================================

class FBcprAuxModel(FBcprModel):
    """
    FBcprAux 모델 (FBcpr + Auxiliary Critic)

    추가 구성요소:
    - aux_critic: 보조 보상용 Q-function
    - aux_reward_normalizer: 보조 보상 정규화

    보조 보상 목록:
    - penalty_action_rate: 액션 변화율 페널티
    - penalty_feet_ori: 발 방향 페널티
    - penalty_ankle_roll: 발목 롤 페널티
    - limits_dof_pos: 관절 한계 페널티
    - penalty_slippage: 미끄러짐 페널티
    - penalty_undesired_contact: 원치않는 접촉 페널티
    """
    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, device: str = "cuda"):
        super().__init__(obs_dim, z_dim, action_dim, device)

        # Auxiliary Critic: 보조 보상의 Q-value
        self._aux_critic = ForwardMap(obs_dim, z_dim, action_dim, output_dim=1)

        # 보조 보상 정규화
        self._aux_reward_normalizer = RewardNormalizer()

        self.to(device)

    def _prepare_for_train(self) -> None:
        """Target 네트워크 생성"""
        super()._prepare_for_train()
        self._target_aux_critic = copy.deepcopy(self._aux_critic)


# =============================================================================
# 6. FBAgent - 기본 FB 에이전트
# =============================================================================

class FBAgent:
    """
    Forward-Backward 에이전트

    학습 구성요소:
    - forward_optimizer: Forward Map 학습
    - backward_optimizer: Backward Map 학습
    - actor_optimizer: Actor 학습

    학습 알고리즘:
    1. FB Loss: Successor Feature 학습
    2. Actor Loss: FB Q-value 최대화
    """

    def __init__(self, model: FBModel, cfg):
        self.cfg = cfg
        self._model = model
        self.device = model.device

        # 학습 설정
        self.fb_target_tau = 0.01  # Target 네트워크 업데이트 비율

        self.setup_training()

    def setup_training(self) -> None:
        """학습 설정"""
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model._prepare_for_train()

        # 옵티마이저
        self.backward_optimizer = torch.optim.Adam(
            self._model._backward_map.parameters(), lr=1e-5
        )
        self.forward_optimizer = torch.optim.Adam(
            self._model._forward_map.parameters(), lr=3e-4
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(), lr=3e-4
        )

        # 파라미터 리스트 (soft update용)
        self._forward_map_paramlist = tuple(self._model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(self._model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(self._model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(self._model._target_backward_map.parameters())

        # Off-diagonal 마스크 (FB Loss용)
        batch_size = 1024
        self.off_diag = 1 - torch.eye(batch_size, batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

        # Z 버퍼 (혼합 롤아웃용)
        self.z_buffer = ZBuffer(8192, self._model.z_dim, self.device)

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        """액션 선택"""
        return self._model.act(obs, z, mean)

    # -------------------------------------------------------------------------
    # 핵심: Forward-Backward Loss
    # -------------------------------------------------------------------------

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward-Backward Loss 계산 및 업데이트

        핵심 아이디어:
        - Successor Feature Matrix M = F · B^T
        - M[i,j] = s_i에서 시작하여 goal=s_j에 도달할 확률
        - 대각선(M[i,i])은 1에 가깝게, 비대각선은 0에 가깝게

        Loss 구성:
        1. FB Loss (Bellman Consistency):
           - Target: M_target = γ * F_target(s', z, a') · B_target(g)^T
           - Loss: ||M - Target||^2

        2. Orthonormality Loss:
           - B(s)가 직교 정규 기저를 형성하도록
           - Cov = B · B^T, Loss = ||Cov - I||^2

        수학적 배경:
        - Successor Feature: ψ(s,a) = E[Σ γ^t φ(s_t)]
        - FB 분해: ψ(s,a) = F(s,a) · B(g)
        - M(s,g) = F(s,a) · B(g)^T ≈ γ^T (기대 할인 도달)
        """
        with torch.no_grad():
            # Target 계산
            # 다음 상태에서 정책으로 액션 선택
            next_action = self.sample_action_from_norm_obs(next_obs, z)

            # Target Successor Feature
            target_Fs = self._model._target_forward_map(next_obs, z, next_action)  # [num_parallel, batch, z_dim]
            target_B = self._model._target_backward_map(goal)  # [batch, z_dim]

            # Target M = F_target · B_target^T
            target_Ms = torch.matmul(target_Fs, target_B.T)  # [num_parallel, batch, batch]

            # 불확실성 고려 (pessimism)
            _, _, target_M = self.get_targets_uncertainty(target_Ms, pessimism_penalty=0.0)  # [batch, batch]

        # 현재 네트워크로 계산
        Fs = self._model._forward_map(obs, z, action)  # [num_parallel, batch, z_dim]
        B = self._model._backward_map(goal)  # [batch, z_dim]
        Ms = torch.matmul(Fs, B.T)  # [num_parallel, batch, batch]

        # FB Loss
        diff = Ms - discount * target_M  # Bellman error

        # Off-diagonal Loss: M[i,j] (i≠j) → 0
        # "다른 상태에 도달하지 않도록"
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum

        # Diagonal Loss: M[i,i] → 1
        # "자기 자신(목표)에 도달하도록"
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]

        fb_loss = fb_offdiag + fb_diag

        # Orthonormality Loss
        # B가 직교 정규 기저를 형성하도록
        # Cov = B · B^T ≈ I
        Cov = torch.matmul(B, B.T)  # [batch, batch]

        # 대각선: ||diag(Cov)|| → 1
        orth_loss_diag = -Cov.diag().mean()

        # 비대각선: ||off_diag(Cov)|| → 0
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum

        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += 100.0 * orth_loss  # ortho_coef = 100

        # 최적화
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        return {
            "fb_loss": fb_loss.detach(),
            "fb_diag": fb_diag.detach(),
            "fb_offdiag": fb_offdiag.detach(),
            "orth_loss": orth_loss.detach(),
        }

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Actor 업데이트

        목표: Q(s, z, a) 최대화

        Q(s, z, a) = F(s, z, a) · z

        Loss = -E[Q(s, z, π(s, z))]
        """
        # 정책으로 액션 샘플링
        dist = self._model._actor(obs, z, std=0.05)
        action = dist.sample(clip=0.3)  # stddev_clip

        # Q-value 계산: F · z
        Fs = self._model._forward_map(obs, z, action)  # [num_parallel, batch, z_dim]
        Qs = (Fs * z).sum(dim=-1)  # [num_parallel, batch]

        # 불확실성 고려 (pessimism)
        _, _, Q = self.get_targets_uncertainty(Qs, pessimism_penalty=0.5)  # [batch]

        actor_loss = -Q.mean()

        # 최적화
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "Q_fb": Q.mean().detach()}

    def get_targets_uncertainty(
        self,
        preds: torch.Tensor,
        pessimism_penalty: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        앙상블 예측에서 불확실성 추정

        Args:
            preds: 앙상블 예측 [num_parallel, batch, ...]
            pessimism_penalty: 비관적 페널티 계수

        Returns:
            mean: 평균 예측
            uncertainty: 예측 불확실성
            target: 비관적 목표값 (mean - penalty * uncertainty)

        이유:
        - 앙상블 간 불일치가 크면 불확실성이 높음
        - 불확실할 때 보수적으로 행동 (pessimism)
        """
        preds_mean = preds.mean(dim=0)

        # 앙상블 간 차이로 불확실성 계산
        preds_uns = preds.unsqueeze(dim=0)  # [1, num_parallel, ...]
        preds_uns2 = preds.unsqueeze(dim=1)  # [num_parallel, 1, ...]
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # [num_parallel, num_parallel, ...]

        num_parallel = preds.shape[0]
        num_parallel_scaling = num_parallel ** 2 - num_parallel
        preds_unc = preds_diffs.sum(dim=(0, 1)) / num_parallel_scaling

        # 비관적 목표값
        target = preds_mean - pessimism_penalty * preds_unc

        return preds_mean, preds_unc, target

    def sample_action_from_norm_obs(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """정규화된 obs에서 액션 샘플링"""
        dist = self._model._actor(obs, z, std=0.05)
        return dist.sample(clip=0.3)


# =============================================================================
# 7. FBcprAgent - Discriminator 추가
# =============================================================================

class FBcprAgent(FBAgent):
    """
    FBcpr 에이전트 (FB + Discriminator + Critic)

    추가 학습:
    - Discriminator: 전문가/정책 판별
    - Critic: 판별자 보상의 Q-value

    학습 흐름:
    1. 전문가 z 인코딩
    2. 판별자 업데이트 (GAN)
    3. z 혼합 샘플링
    4. FB 업데이트
    5. Critic 업데이트
    6. Actor 업데이트
    """

    def setup_training(self) -> None:
        """학습 설정"""
        super().setup_training()

        # 추가 파라미터 리스트
        self._critic_map_paramlist = tuple(self._model._critic.parameters())
        self._target_critic_map_paramlist = tuple(self._model._target_critic.parameters())

        # 추가 옵티마이저
        self.critic_optimizer = torch.optim.Adam(
            self._model._critic.parameters(), lr=3e-4
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self._model._discriminator.parameters(), lr=1e-5
        )

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        """
        전체 업데이트 루프

        1. 배치 샘플링
        2. 전문가 z 인코딩
        3. 판별자 업데이트
        4. z 혼합 샘플링
        5. z 리라벨링
        6. FB 업데이트
        7. Critic 업데이트
        8. Actor 업데이트
        9. Target 네트워크 업데이트
        """
        batch_size = 1024

        # 1. 배치 샘플링
        expert_batch = replay_buffer["expert_slicer"].sample(batch_size)
        train_batch = replay_buffer["train"].sample(batch_size)

        # 데이터 추출
        train_obs = train_batch["observation"]
        train_action = train_batch["action"]
        train_next_obs = train_batch["next"]["observation"]
        discount = 0.98 * ~train_batch["next"]["terminated"]  # γ * (1 - done)

        expert_obs = expert_batch["observation"]
        expert_next_obs = expert_batch["next"]["observation"]

        # 관측값 정규화
        self._model._obs_normalizer(train_obs)  # 통계 업데이트

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs = self._model._obs_normalizer(train_obs)
            train_next_obs = self._model._obs_normalizer(train_next_obs)
            expert_obs = self._model._obs_normalizer(expert_obs)
            expert_next_obs = self._model._obs_normalizer(expert_next_obs)

        # 2. 전문가 z 인코딩
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"]

        # 3. 판별자 업데이트
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=10.0  # WGAN-GP
        )

        # 4. z 혼합 샘플링
        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z)
        self.z_buffer.add(z)  # Z 버퍼에 추가

        # 5. z 리라벨링 (80% 확률)
        relabel_ratio = 0.8
        mask = torch.rand((batch_size, 1), device=self.device) <= relabel_ratio
        train_z = torch.where(mask, z, train_z)

        # 6. FB 업데이트
        metrics.update(self.update_fb(
            obs=train_obs,
            action=train_action,
            discount=discount,
            next_obs=train_next_obs,
            goal=train_next_obs,
            z=train_z,
        ))

        # 7. Critic 업데이트
        metrics.update(self.update_critic(
            obs=train_obs,
            action=train_action,
            discount=discount,
            next_obs=train_next_obs,
            z=train_z,
        ))

        # 8. Actor 업데이트
        metrics.update(self.update_actor(
            obs=train_obs,
            z=train_z,
        ))

        # 9. Target 네트워크 soft update
        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, 0.01)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, 0.01)
            _soft_update_params(self._critic_map_paramlist, self._target_critic_map_paramlist, 0.005)

        return metrics

    @torch.no_grad()
    def encode_expert(self, next_obs: torch.Tensor) -> torch.Tensor:
        """
        전문가 궤적 인코딩

        Args:
            next_obs: 전문가 관측값 [batch, obs_dim]

        Returns:
            z_expert: 전문가 z [batch, z_dim]

        알고리즘:
        1. 각 관측값에 대해 B(obs) 계산
        2. seq_length 단위로 그룹핑
        3. 그룹 내 평균으로 z 계산
        4. 그룹 내 모든 샘플에 같은 z 할당
        """
        batch_size = 1024
        seq_length = 8

        B_expert = self._model._backward_map(next_obs)  # [batch, z_dim]

        # [batch, z_dim] → [N, seq_length, z_dim]
        B_expert = B_expert.view(
            batch_size // seq_length,
            seq_length,
            B_expert.shape[-1]
        )

        # 시퀀스 내 평균
        z_expert = B_expert.mean(dim=1)  # [N, z_dim]
        z_expert = self._model.project_z(z_expert)

        # 다시 배치 크기로 확장
        z_expert = torch.repeat_interleave(z_expert, seq_length, dim=0)  # [batch, z_dim]

        return z_expert

    @torch.no_grad()
    def sample_mixed_z(
        self,
        train_goal: torch.Tensor,
        expert_encodings: torch.Tensor
    ) -> torch.Tensor:
        """
        혼합 z 분포 샘플링

        세 가지 소스에서 z 샘플링:
        1. 목표 인코딩 (train_goal_ratio = 0.2)
        2. 전문가 궤적 인코딩 (expert_asm_ratio = 0.6)
        3. 균등 분포 (나머지 0.2)

        이유:
        - 다양한 z 분포로 일반화 능력 향상
        - 전문가 z는 실제 태스크 분포 반영
        - 목표 z는 버퍼 내 다양성 활용
        - 균등 z는 탐색 촉진
        """
        batch_size = 1024
        p_goal = 0.2
        p_expert_asm = 0.6

        # 기본: 균등 분포
        z = self._model.sample_z(batch_size, device=self.device)

        # 소스 선택 (확률적)
        prob = torch.tensor([p_goal, p_expert_asm, 1 - p_goal - p_expert_asm], device=self.device)
        mix_idxs = torch.multinomial(prob, num_samples=batch_size, replacement=True).reshape(-1, 1)

        # 목표 인코딩 (20%)
        perm = torch.randperm(batch_size, device=self.device)
        train_goal_shuffled = train_goal[perm]
        goals = self._model._backward_map(train_goal_shuffled)
        goals = self._model.project_z(goals)
        z = torch.where(mix_idxs == 0, goals, z)

        # 전문가 인코딩 (60%)
        perm = torch.randperm(batch_size, device=self.device)
        z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

        return z

    def update_discriminator(
        self,
        expert_obs: torch.Tensor,
        expert_z: torch.Tensor,
        train_obs: torch.Tensor,
        train_z: torch.Tensor,
        grad_penalty: float,
    ) -> Dict[str, torch.Tensor]:
        """
        판별자 업데이트 (GAN)

        목표:
        - 전문가: D(s, z) → 1
        - 정책: D(s, z) → 0

        Loss:
        - Binary Cross Entropy
        - expert_loss = -log(D(expert))
        - policy_loss = -log(1 - D(policy)) = softplus(D(policy))
        - loss = expert_loss + policy_loss

        Regularization:
        - WGAN Gradient Penalty: ||∇D||² ≈ 1
        """
        # Logits 계산
        expert_logits = self._model._discriminator.compute_logits(expert_obs, expert_z)
        unlabeled_logits = self._model._discriminator.compute_logits(train_obs, train_z)

        # Binary Cross Entropy
        expert_loss = -F.logsigmoid(expert_logits)  # -log(sigmoid(x)) = -log(D)
        unlabeled_loss = F.softplus(unlabeled_logits)  # log(1 + exp(x)) ≈ -log(1-D)
        loss = torch.mean(expert_loss + unlabeled_loss)

        # WGAN Gradient Penalty
        if grad_penalty > 0:
            wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
            loss += grad_penalty * wgan_gp

        # 최적화
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_optimizer.step()

        return {
            "disc_loss": loss.detach(),
            "disc_expert_loss": expert_loss.mean().detach(),
            "disc_train_loss": unlabeled_loss.mean().detach(),
        }

    def gradient_penalty_wgan(
        self,
        real_obs: torch.Tensor,
        real_z: torch.Tensor,
        fake_obs: torch.Tensor,
        fake_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        WGAN Gradient Penalty

        ||∇_x D(x)||² should be close to 1

        x = α * real + (1-α) * fake
        GP = (||∇_x D(x)|| - 1)²

        이유:
        - Lipschitz 제약 강제
        - 학습 안정성 향상
        """
        batch_size = real_obs.shape[0]
        alpha = torch.rand(batch_size, 1, device=self.device)

        # 보간
        interpolated_obs = (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True)
        interpolated_z = (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True)

        # 판별자 출력
        d_interpolates = self._model._discriminator.compute_logits(interpolated_obs, interpolated_z)

        # 그래디언트 계산
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=[interpolated_obs, interpolated_z],
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )

        # GP = (||∇|| - 1)²
        cat_gradients = torch.cat([g.flatten(1) for g in gradients], dim=1)
        gradient_penalty = ((cat_gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Critic 업데이트 (TD Learning)

        목표: Q(s, z, a) ≈ r + γ Q(s', z, a')

        여기서 r = 판별자 보상 (log odds ratio)

        Loss: ||Q - (r + γ Q_target)||²
        """
        num_parallel = 2  # 앙상블 크기

        with torch.no_grad():
            # 판별자 보상
            reward = self._model._discriminator.compute_reward(obs, z)

            # 다음 상태에서 정책으로 액션 선택
            dist = self._model._actor(next_obs, z, std=0.05)
            next_action = dist.sample(clip=0.3)

            # Target Q-value
            next_Qs = self._model._target_critic(next_obs, z, next_action)
            Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, pessimism_penalty=0.5)

            # TD Target
            target_Q = reward + discount * next_V
            expanded_targets = target_Q.expand(num_parallel, -1, -1)

        # 현재 Q-value
        Qs = self._model._critic(obs, z, action)

        # MSE Loss
        critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # 최적화
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": critic_loss.detach(),
            "mean_disc_reward": reward.mean().detach(),
        }

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Actor 업데이트 (FBcpr)

        목표: Q_disc + Q_fb 최대화

        Q_disc = Critic(s, z, a)  (판별자 보상)
        Q_fb = F(s, z, a) · z     (FB 보상)

        Loss = -Q_disc * reg_coeff - Q_fb
        """
        reg_coeff = 0.05
        scale_reg = True

        # 정책으로 액션 샘플링
        dist = self._model._actor(obs, z, std=0.05)
        action = dist.sample(clip=0.3)

        # 판별자 보상 Q-value
        Qs_discriminator = self._model._critic(obs, z, action)
        _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, pessimism_penalty=0.5)

        # FB 보상 Q-value
        Fs = self._model._forward_map(obs, z, action)
        Qs_fb = (Fs * z).sum(dim=-1)
        _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, pessimism_penalty=0.5)

        # 가중치 스케일링
        weight = Q_fb.abs().mean().detach() if scale_reg else 1.0

        # Actor Loss
        actor_loss = -Q_discriminator.mean() * reg_coeff * weight - Q_fb.mean()

        # 최적화
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "Q_discriminator": Q_discriminator.mean().detach(),
            "Q_fb": Q_fb.mean().detach(),
        }


# =============================================================================
# 8. FBcprAuxAgent - Auxiliary Rewards 추가
# =============================================================================

class FBcprAuxAgent(FBcprAgent):
    """
    FBcprAux 에이전트 (FBcpr + Auxiliary Rewards)

    추가 학습:
    - aux_critic: 보조 보상의 Q-value

    보조 보상:
    - 안전하고 자연스러운 동작을 유도하는 페널티들

    Actor Loss:
    = -Q_disc * reg_coeff - Q_aux * reg_coeff_aux - Q_fb
    """

    def setup_training(self) -> None:
        """학습 설정"""
        super().setup_training()

        # 추가 파라미터 리스트
        self._aux_critic_map_paramlist = tuple(self._model._aux_critic.parameters())
        self._aux_target_critic_map_paramlist = tuple(self._model._target_aux_critic.parameters())

        # 추가 옵티마이저
        self.aux_critic_optimizer = torch.optim.Adam(
            self._model._aux_critic.parameters(), lr=3e-4
        )

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        """
        전체 업데이트 루프 (FBcprAux)

        FBcprAgent.update() + aux_critic 업데이트
        """
        batch_size = 1024

        # FBcprAgent와 동일한 전처리
        expert_batch = replay_buffer["expert_slicer"].sample(batch_size)
        train_batch = replay_buffer["train"].sample(batch_size)

        train_obs = train_batch["observation"]
        train_action = train_batch["action"]
        train_next_obs = train_batch["next"]["observation"]
        discount = 0.98 * ~train_batch["next"]["terminated"]

        expert_obs = expert_batch["observation"]
        expert_next_obs = expert_batch["next"]["observation"]

        self._model._obs_normalizer(train_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs = self._model._obs_normalizer(train_obs)
            train_next_obs = self._model._obs_normalizer(train_next_obs)
            expert_obs = self._model._obs_normalizer(expert_obs)
            expert_next_obs = self._model._obs_normalizer(expert_next_obs)

        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"]

        # 판별자 업데이트
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=10.0
        )

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z)
        self.z_buffer.add(z)

        mask = torch.rand((batch_size, 1), device=self.device) <= 0.8
        train_z = torch.where(mask, z, train_z)

        # FB 업데이트
        metrics.update(self.update_fb(
            obs=train_obs,
            action=train_action,
            discount=discount,
            next_obs=train_next_obs,
            goal=train_next_obs,
            z=train_z,
        ))

        # Critic 업데이트
        metrics.update(self.update_critic(
            obs=train_obs,
            action=train_action,
            discount=discount,
            next_obs=train_next_obs,
            z=train_z,
        ))

        # =====================================================================
        # 추가: Auxiliary Critic 업데이트
        # =====================================================================

        # 보조 보상 계산 (가중 합)
        aux_rewards = {
            'penalty_action_rate': -0.1,
            'penalty_feet_ori': -0.4,
            'penalty_ankle_roll': -4.0,
            'limits_dof_pos': -10.0,
            'penalty_slippage': -2.0,
            'penalty_undesired_contact': -1.0,
        }

        aux_reward = torch.zeros((batch_size, 1), device=self.device)
        for name, scale in aux_rewards.items():
            if name in train_batch.get("aux_rewards", {}):
                aux_reward += scale * train_batch["aux_rewards"][name]

        # 보조 보상 정규화
        aux_reward = self._model._aux_reward_normalizer(aux_reward)

        metrics.update(self.update_aux_critic(
            obs=train_obs,
            action=train_action,
            discount=discount,
            aux_reward=aux_reward,
            next_obs=train_next_obs,
            z=train_z,
        ))

        # Actor 업데이트 (Q_fb + Q_disc + Q_aux)
        metrics.update(self.update_actor_aux(
            obs=train_obs,
            z=train_z,
        ))

        # Target 네트워크 soft update
        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, 0.01)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, 0.01)
            _soft_update_params(self._critic_map_paramlist, self._target_critic_map_paramlist, 0.005)
            _soft_update_params(self._aux_critic_map_paramlist, self._aux_target_critic_map_paramlist, 0.005)

        return metrics

    def update_aux_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        aux_reward: torch.Tensor,
        next_obs: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Auxiliary Critic 업데이트

        목표: Q_aux(s, z, a) ≈ r_aux + γ Q_aux(s', z, a')

        여기서 r_aux = 보조 보상 (페널티 합)
        """
        num_parallel = 2

        with torch.no_grad():
            dist = self._model._actor(next_obs, z, std=0.05)
            next_action = dist.sample(clip=0.3)

            next_Qs = self._model._target_aux_critic(next_obs, z, next_action)
            Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, pessimism_penalty=0.5)

            target_Q = aux_reward + discount * next_V
            expanded_targets = target_Q.expand(num_parallel, -1, -1)

        Qs = self._model._aux_critic(obs, z, action)
        aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        self.aux_critic_optimizer.step()

        return {
            "aux_critic_loss": aux_critic_loss.detach(),
            "mean_aux_reward": aux_reward.mean().detach(),
        }

    def update_actor_aux(self, obs: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Actor 업데이트 (FBcprAux)

        Loss = -Q_disc * reg_coeff * w - Q_aux * reg_coeff_aux * w - Q_fb

        reg_coeff = 0.05 (판별자 보상 가중치)
        reg_coeff_aux = 0.02 (보조 보상 가중치)
        """
        reg_coeff = 0.05
        reg_coeff_aux = 0.02
        scale_reg = True

        # 정책으로 액션 샘플링
        dist = self._model._actor(obs, z, std=0.05)
        action = dist.sample(clip=0.3)

        # 판별자 보상 Q-value
        Qs_discriminator = self._model._critic(obs, z, action)
        _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, pessimism_penalty=0.5)

        # 보조 보상 Q-value
        Qs_aux = self._model._aux_critic(obs, z, action)
        _, _, Q_aux = self.get_targets_uncertainty(Qs_aux, pessimism_penalty=0.5)

        # FB 보상 Q-value
        Fs = self._model._forward_map(obs, z, action)
        Qs_fb = (Fs * z).sum(dim=-1)
        _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, pessimism_penalty=0.5)

        # 가중치 스케일링
        weight = Q_fb.abs().mean().detach() if scale_reg else 1.0

        # Actor Loss
        actor_loss = (
            -Q_discriminator.mean() * reg_coeff * weight
            - Q_aux.mean() * reg_coeff_aux * weight
            - Q_fb.mean()
        )

        # 최적화
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "Q_discriminator": Q_discriminator.mean().detach(),
            "Q_aux": Q_aux.mean().detach(),
            "Q_fb": Q_fb.mean().detach(),
        }


# =============================================================================
# 9. 보조 클래스들
# =============================================================================

class TruncatedNormal:
    """Truncated Normal 분포 (간략화)"""
    def __init__(self, mu: torch.Tensor, std: torch.Tensor):
        self.mu = mu
        self.std = std

    @property
    def mean(self):
        return self.mu

    def sample(self, clip: float = None) -> torch.Tensor:
        noise = torch.randn_like(self.mu) * self.std
        if clip is not None:
            noise = torch.clamp(noise, -clip, clip)
        return self.mu + noise


class ZBuffer:
    """Z 버퍼 (혼합 롤아웃용)"""
    def __init__(self, capacity: int, z_dim: int, device: str):
        self.capacity = capacity
        self.buffer = torch.zeros((capacity, z_dim), device=device)
        self.ptr = 0
        self.size = 0

    def add(self, z: torch.Tensor):
        batch_size = z.shape[0]
        indices = (torch.arange(batch_size, device=z.device) + self.ptr) % self.capacity
        self.buffer[indices] = z
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int, device: str) -> torch.Tensor:
        indices = torch.randint(0, self.size, (batch_size,), device=device)
        return self.buffer[indices]

    def empty(self) -> bool:
        return self.size == 0


class RewardNormalizer(nn.Module):
    """보상 정규화"""
    def __init__(self, translate: bool = False, scale: bool = True):
        super().__init__()
        self.translate = translate
        self.scale = scale
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('var', torch.ones(1))

    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        # Running 통계 업데이트 (간략화)
        if self.training:
            self.mean = 0.99 * self.mean + 0.01 * reward.mean()
            self.var = 0.99 * self.var + 0.01 * reward.var()

        if self.translate:
            reward = reward - self.mean
        if self.scale:
            reward = reward / (self.var.sqrt() + 1e-8)

        return reward


class DenseParallel(nn.Module):
    """앙상블용 병렬 Linear 레이어 (간략화)"""
    def __init__(self, in_features: int, out_features: int, n_parallel: int):
        super().__init__()
        self.n_parallel = n_parallel
        self.weight = nn.Parameter(torch.randn(n_parallel, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(n_parallel, 1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n_parallel, batch, in_features]
        return torch.bmm(x, self.weight) + self.bias


class ParallelLayerNorm(nn.Module):
    """앙상블용 병렬 LayerNorm (간략화)"""
    def __init__(self, normalized_shape: list, n_parallel: int):
        super().__init__()
        self.n_parallel = n_parallel
        self.layer_norms = nn.ModuleList([nn.LayerNorm(normalized_shape) for _ in range(n_parallel)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n_parallel, batch, features]
        outputs = [ln(x[i]) for i, ln in enumerate(self.layer_norms)]
        return torch.stack(outputs, dim=0)


# =============================================================================
# 10. 알고리즘 요약
# =============================================================================
"""
FBcpr 알고리즘 요약
==================

1. Forward-Backward 표현 학습
   - B(s): 상태 → 잠재 벡터 z
   - F(s, z, a): Successor Feature 예측
   - M = F · B^T: 상태-목표 도달 행렬

2. 판별자 학습 (GAN)
   - D(s, z): 전문가 vs 정책 판별
   - 암묵적 보상: r = log D - log(1-D)
   - WGAN-GP로 안정화

3. Z 분포 학습
   - 목표 인코딩: z = B(goal)
   - 전문가 인코딩: z = mean(B(expert_traj))
   - 혼합 분포: 20% 목표 + 60% 전문가 + 20% 균등

4. Actor 학습
   - Q_fb = F(s, z, a) · z
   - Q_disc = Critic(s, z, a)
   - Q_aux = AuxCritic(s, z, a)
   - Loss = -Q_disc * 0.05 - Q_aux * 0.02 - Q_fb

5. 추론
   - 모션 트래킹: z = mean(B(traj[t:t+8]))
   - 목표 도달: z = B(goal_frame)
   - 보상 최적화: z = Σ r_i * B(s_i)

핵심 인사이트:
- z는 목표/태스크를 나타내는 조건 변수
- 같은 정책 π(s, z)가 다른 z로 다른 행동
- 판별자로 "전문가처럼" 행동하도록 유도
- 보조 보상으로 안전성 확보
"""
