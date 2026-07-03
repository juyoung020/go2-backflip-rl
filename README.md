# Go2 Backflip RL 🐕‍🦺

**Unitree Go2 로봇개에게 백덤블링(backflip)을 가르치는 강화학습 프로젝트.**
참조 모션 없이(reference-free) 보상 설계와 커리큘럼만으로 학습하고, Isaac Sim → MuJoCo → (실기) 파이프라인으로 검증합니다.

<p align="center">
  <img src="media/preview.gif" alt="Go2 백플립 — 4마리 동시 시연 (최종 정책)" width="640">
</p>

<p align="center">
  <a href="media/go2_backflip_연착지_최종.mp4">🎬 원본 영상(mp4)</a> ·
  <a href="docs/go2_backflip_발표자료_v2.html">📊 발표자료</a>
</p>

## 최종 성능 (MuJoCo 실측)

| 지표 | 값 |
|---|---|
| 회전각 | **359.5°** (오차 0.081 rad) |
| 착지 충격 | **194 N = 체중 1.3배** (초기 대비 −59%) |
| 착지 위치 오차 | **0.094 m** (초기 0.88 m → −89%) |
| 점프 최고점 | 0.78 m |
| 착지 자세 | 완전 수평 (g_z −1.00) |

두 개의 독립된 물리 엔진(Isaac Sim/PhysX, MuJoCo)에서 모두 검증되었습니다.

## 파이프라인

```
① Isaac Sim/IsaacLab (PhysX)          ② MuJoCo                    ③ 실기 Go2
   8,192 병렬 환경, 111k steps/s          zero-shot 검증(PASS)         unitree_sdk2 LowCmd
   5단계 커리큘럼, ~13억 스텝       →     + fine-tune 재학습     →     (동일 PD 인터페이스)
   보상 16항 · asymmetric AC             착지 정밀도·연착지 개선        [준비 완료]
```

## 데모 영상 (`media/`)

| 영상 | 내용 |
|---|---|
| `stage2_180도회전.mp4` | 커리큘럼 중간 — 후방 180° 회전 |
| `풀백플립_성공.mp4` | Stage 3 — 첫 360° 백플립 |
| `최종_강건화.mp4` | Stage 4 — 도메인 랜덤화 하 백플립 |
| `mujoco_sim2sim.mp4` | MuJoCo zero-shot 재현 |
| `mujoco_재학습.mp4` | MuJoCo fine-tune 후 |
| `연착지_최종.mp4` | **최종** — 사뿐한 제자리 백플립 |

## 핵심 설계

### MDP 정식화
- **관측 (정책, 43차원)**: 각속도 3 + 중력벡터 3 + 관절각 12 + 관절속도 12 + 이전행동 12 + **phase 1** — 실기 센서(IMU+엔코더)로 100% 재현 가능한 정보만 사용
- **관측 (크리틱, 51차원)**: + 선속도·발접촉·높이 (특권 관측, asymmetric actor-critic)
- **phase 변수가 핵심**: 시간 구간별 보상이 상태의 함수 R(s,a)가 되어 Markov 성질 보존 → LSTM 없이 MLP [512,256,128]로 충분
- **행동**: 관절 목표각 오프셋 12차원, 50Hz → PD(Kp 25, Kd 0.5) + DC모터 속도-토크 곡선, 200Hz

### 보상함수 (16항)
| 그룹 | 항목 |
|---|---|
| 태스크(+) | 도약 상승속도(클램프) · 후방 회전 각속도 · 착지 성공 보너스 |
| 자세(−) | 피치 램프 추종(0→−2π) · 높이 · 롤/요 억제 · 좌우대칭 · 조기점프 방지 · 스탠스 폭 |
| 품질(−) | 토크² · 관절가속² · 행동변화율 · 관절한계 · 몸통접촉 · **착지충격 초과분** · **제자리 이탈²** |

가중치 출발점은 실기 실증 사례([Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip))이며, 전 항목이 학습 도중 `reward_overrides.json` 편집으로 **실시간 변경** 가능합니다 (에피소드 리셋마다 파일 감시).

### 커리큘럼 (5단계, 조기 승급)
| Stage | 내용 | 승급 기준 (실측 물리량, 3회 연속) |
|---|---|---|
| 1 | 제자리 수직 점프 | 도약 보상 ≥ 5.0 → **494 iter에서 승급** |
| 2 | 후방 180° (등 착지 허용) | 회전각 ≤ −2.8 rad → **−3.13 (오차 0.2%)** |
| 3 | 풀 백플립 360° | 회전각 ≤ −5.9 rad → **−6.28 = 359.96°** |
| 4 | 강건화 (DR: 마찰·질량·무게중심·게인) | 완주 |
| 5 | 정밀·연착지 (제자리 + 충격 흡수) | 완주 + 실측 반복 조정 |

각 스테이지는 이전 체크포인트(정책+Adam 상태)를 warm-start하며, 보상 항은 고정하고 가중치·커리큘럼 변수만 전환합니다.

## 빠른 시작

### 1. Isaac Sim 학습 (GPU)
```bash
# 요구: Isaac Sim 4.5 + IsaacLab 2.3 (경로는 환경에 맞게)
ISAAC_PY=~/isaac-sim-4.5.0/python.sh
$ISAAC_PY -m pip install -e .

$ISAAC_PY scripts/rsl_rl/train.py --task Isaac-Backflip-Go2-Stage1-Direct-v0 \
  --headless --num_envs 8192 --max_iterations 1000
# Stage 2~5는 --resume으로 이어서 (자세한 명령은 아래 태스크 ID 표 참고)
```

등록 태스크: `Isaac-Backflip-Go2-Stage{1..6}-Direct-v0` (Stage 6 = CMDP 충격 하드제약, 선택)

### 2. MuJoCo 검증·재학습 (CPU)
```bash
python3 -m venv .venv-mj
.venv-mj/bin/pip install mujoco onnxruntime "imageio[ffmpeg]" numpy tensorboard rsl-rl-lib==3.0.1 tensordict
.venv-mj/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set unitree_go2 && cd ..

# zero-shot 검증 (제공된 최종 정책으로 바로 실행 가능)
MUJOCO_GL=egl .venv-mj/bin/python scripts/sim2sim_mujoco.py \
  --onnx policy/policy_soft3.onnx --video backflip.mp4

# fine-tune 재학습
.venv-mj/bin/python scripts/mujoco_finetune.py \
  --checkpoint policy/model_6594.pt --num_envs 128 --iterations 300
```

### 3. 프레임 단위 실측 (보상 튜닝용)
```bash
$ISAAC_PY scripts/rsl_rl/capture.py --task Isaac-Backflip-Go2-Stage5-Direct-v0 \
  --headless --num_envs 4 --steps 200 --checkpoint <ckpt> --out capture.csv
# 50Hz로 높이·각속도·발접촉·토크 기록 + phase 타이밍 요약 출력
```

## 저장소 구조

```
├── source/go2_backflip/          # IsaacLab 외부 확장 (환경·보상·커리큘럼 cfg)
│   └── .../go2_backflip_env.py   #   16항 보상, 관측 43/51, 지연·노이즈·리셋 DR
├── scripts/
│   ├── rsl_rl/train.py, play.py  # 학습/재생 (IsaacLab 기반)
│   ├── rsl_rl/capture.py         # 프레임 실측 도구
│   ├── sim2sim_mujoco.py         # MuJoCo zero-shot 검증 (+영상)
│   ├── mujoco_finetune.py        # MuJoCo 재학습 (rsl_rl VecEnv 포팅)
│   └── export_mj_onnx.py         # 체크포인트 → ONNX
├── policy/                       # 최종 정책 (ONNX + 체크포인트)
├── media/                        # 데모 영상 6종
├── docs/                         # 발표자료 (HTML 슬라이드)
└── reward_overrides.json         # 실시간 보상 튜닝 파일 (런마다 재생성)
```

## sim2sim에서 배운 것 (삽질 로그)

1. **menagerie go2.xml은 관절 감쇠 2.0 + 마찰손실 0.2가 내장** — 학습 조건(감쇠 0, PD Kd만)과 달라 정책이 시동조차 안 걸림. 반드시 0으로 정렬할 것.
2. **토크를 상수로 클립하면 과회전(534°)** — IsaacLab DCMotor의 속도-토크 곡선(속도↑→가용토크↓)까지 이식해야 에너지가 맞음.
3. **Isaac 관절 순서는 type-major** `[FL,FR,RL,RR]×[hip,thigh,calf]` — MuJoCo(leg-major)와 다르므로 이름 기반 매핑 필수.
4. **순간 이벤트 페널티(착지 충격)는 지속 보상(제자리 유지)에 밀린다** — 가중치를 실측 기반으로 수십 배 올려야 균형이 잡혔음.
5. 검증 프로세스가 학습 프로세스의 실시간 튜닝 파일을 덮어쓰지 않게 **소유권 격리** 필요 (실제 사고 후 가드 추가).

## 크레딧

- 보상 설계 기반: [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip) (Go2 백플립 실기 실증)
- 커리큘럼 근거: [Curriculum-Based RL for Quadrupedal Jumping](https://arxiv.org/abs/2401.16337)
- 시뮬레이터: [IsaacLab](https://github.com/isaac-sim/IsaacLab) · [MuJoCo](https://mujoco.org) · [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- RL: [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
