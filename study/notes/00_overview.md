# BFM-Zero 코드베이스 개요

## 프로젝트 소개

BFM-Zero (Behavioral Foundation Model)는 비지도 강화학습을 사용한 휴머노이드 로봇 제어를 위한 행동 기반 모델입니다.

### 핵심 기능
1. **Motion Tracking**: 전문가 모션 데이터를 따라가는 정책 학습
2. **Goal Reaching**: 목표 자세로 이동하는 정책
3. **Reward-based Tasks**: 보상 함수 기반 태스크 수행

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                        train.py                              │
│                      (TrainConfig)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │  Agent   │   │   Env    │   │   Buffer     │
    │ (FBcpr)  │   │ (Isaac)  │   │ (Trajectory) │
    └────┬─────┘   └────┬─────┘   └──────────────┘
         │              │
         ▼              ▼
    ┌──────────┐   ┌──────────┐
    │  Model   │   │Simulator │
    │(nn_models)│  │(MuJoCo/  │
    └──────────┘   │ IsaacSim)│
                   └──────────┘
```

## 학습 파이프라인

1. **환경 초기화**: `HumanoidVerseIsaacConfig`로 시뮬레이션 환경 설정
2. **버퍼 생성**: 전문가 데이터 + 온라인 수집 데이터
3. **에이전트 학습**: FBcpr 알고리즘으로 정책 업데이트
4. **평가**: 모션 트래킹 EMD(Earth Mover's Distance) 측정

## 추론 파이프라인

1. **체크포인트 로드**: `load_model_from_checkpoint_dir()`
2. **환경 설정**: Hydra 오버라이드로 추론용 설정
3. **Latent z 계산**: 태스크에 따라 다른 방식으로 계산
4. **ONNX 내보내기**: 실제 로봇 배포용

## 스터디 노트 목차

| # | 파일 | 주제 |
|---|------|------|
| 01 | `01_macos_installation.md` | macOS 설치 및 문제 해결 |
| 02 | `02_train_analysis.md` | train.py 상세 분석 |
| 03 | `03_fbcpr_algorithm.md` | FB-CPR 알고리즘 (코드 중심) |
| 04 | `04_environment_config.md` | 환경 설정 시스템 |
| 05 | `05_inference_scripts.md` | 추론 스크립트 분석 |
| 06 | `06_neural_network_architecture.md` | 신경망 구조 |
| 07 | `07_learning_algorithms.md` | 학습 알고리즘 (수학/텐서 흐름) |
| 08 | `08_replay_buffers.md` | 리플레이 버퍼 |
| 09 | `09_evaluation_system.md` | 평가 시스템 (EMD) |
| 10 | `10_env_simulators.md` | 환경 & 시뮬레이터 |
| 11 | `11_motion_library.md` | 모션 라이브러리 |
| 12 | `12_inference_advanced.md` | Goal/Reward Inference 심화 |
| 13 | `13_practical_exercises.md` | 실습 가이드 & 스크립트 목록 |

## 실습 스크립트 (`study/scripts/`)

모델이 필요한 스크립트: `tracking_inference_macos.py`, `goal_inference_macos.py`, `friction_experiment.py`, `analyze_z_vectors.py`

CPU 단독 실행 가능: `debug_tensor_shapes.py`, `model_architecture_compare.py`, `mujoco_viewer_test.py`

로그/결과 파일 필요: `visualize_training.py`, `compare_reward_z.py`
