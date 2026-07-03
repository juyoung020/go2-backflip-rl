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

### 에피소드 구조 — 2초짜리 각본

보상은 phase 시간 창(window)으로 구간화됩니다. 에피소드는 50Hz × 100스텝 = 2.0초:

```
0.0s ──────── 0.5s ── 0.75s ── 1.0s ──────────────── 2.0s
│   준비(웅크림)  │ 도약 │  회전   │    착지·자세 회복      │
│   Φ_prep      │ Φ_up │ Φ_flip  │       Φ_land          │
```

phase 변수 φ = t/T가 관측에 포함되므로 시간 의존 보상이 R(s,a)로 표현됩니다 (Markov 성질 보존).

### 보상함수 — R(s,a) = Σ wᵢ·rᵢ, 16항 전체

**태스크 보상 (하게 만드는 힘)**
| 항 | 수식 | 가중치 |
|---|---|---|
| `r_up` 도약 | `clip(v_z, 0, 3.0) · 𝟙[φ∈Φ_up]` | +20.0 |
| `r_flip` 회전 | `clip(−ω_y, 0, ω_max) · 𝟙[φ∈Φ_flip]` — ω_max는 커리큘럼 변수 | +5.0 |
| `r_land` 착지 보너스 | `𝟙[네발접지 ∧ 수평 ∧ |pitch_int+2π|<0.3] · 𝟙[φ∈Φ_land]` | +10.0 (S4~) |

**자세 유도 (음수)**
| 항 | 수식 | 가중치 |
|---|---|---|
| `r_ori` 피치 추종 | `−(pitch_int − θ_ref(φ))²`, θ_ref: 0→−2π 램프 | −1.0 |
| `r_height` | `−\|h − 0.3\| · 𝟙[φ∉Φ_flip]` | −10.0 |
| `r_roll` / `r_yaw` | `−\|g_y\|` / `−ω_z²` | −10.0 / −1.0 |
| `r_feet_pre` 조기점프 방지 | `−Σ clip(h_feet−0.03, 0) · 𝟙[φ∈Φ_prep]` | −30.0 |
| `r_sym` 좌우대칭 | `−‖a_L·M − a_R‖²` (M: 힙 부호 반전) | −0.1 |
| `r_feet_dist` 스탠스 폭 | `−(\|d_front−0.3\| + \|d_rear−0.3\|)` | −1.0 |

**품질·배포 (음수)**
| 항 | 수식 | 가중치 |
|---|---|---|
| `r_rate` / `r_limit` / `r_contact` | 행동 급변² / 관절 소프트한계 초과 / 몸통·허벅지 접촉 | −0.001 / −10 / −1 |
| `r_torque` / `r_acc` | `−‖τ‖²` / `−‖q̈‖²` | −5e-4 / −1e-6 (S4~) |
| `r_drift` 제자리 | `−‖p_xy − p₀‖² · 𝟙[φ∈Φ_land]` | −5.0 (S5) |
| `r_impact` 연착지 | `−Σ clip(F_feet − 250N, 0)` | −0.005 → **−0.3** (S5, 실측 조정) |

모든 항은 시뮬레이터 **참값 상태**로 계산되며(관측과 무관), 매 스텝 `step_dt`를 곱해 적산합니다. 가중치 출발점은 [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip) 실기 실증값이고, 전 항목이 학습 도중 `reward_overrides.json` 편집으로 **실시간 변경** 가능합니다 (에피소드 리셋마다 파일 감시 → 즉시 반영).

### 커리큘럼 — 보상 항은 고정, 가중치와 변수만 전환

각 스테이지는 이전 체크포인트(정책+Adam 옵티마이저 상태)를 warm-start하고, **무엇이 바뀌는지**는 아래가 전부입니다:

| Stage | 배우는 것 | 바뀌는 설정 | 승급 기준 (실측, 3회 연속) | 실제 결과 |
|---|---|---|---|---|
| **1** | 제자리 수직 점프 | `r_flip`=0, `r_ori`=0, 수평유지 `r_flat`=−5 활성 | 도약 보상 ≥ 5.0 | **494/1000 iter 조기승급** |
| **2** | 후방 180° 회전 | `r_flip`=+5 (ω_max **3.6**), θ_ref→**π**, `r_flat` 제거, 몸통접촉 −1→**−0.1** (등 착지 허용) | 회전각 ≤ −2.8 rad | **−3.13 rad (180.0°)** |
| **3** | 풀 백플립 360° | ω_max **3.6→7.2**, θ_ref→**2π**, 몸통접촉 **−1 복원** (발 착지 강제) | 회전각 ≤ −5.9 rad | **−6.28 rad (359.96°)** |
| **4** | 강건화 | 도메인 랜덤화 ON (마찰 0.5–1.25, 질량 ±1kg, 무게중심 ±3cm, Kp/Kd ±10%) + `r_torque`/`r_acc`/`r_land` 활성 | 조기승급 없음 (완주) | DR 하 360° 유지 |
| **5** | 정밀·연착지 | `r_drift`=−5, `r_impact` 활성, 리셋 랜덤화 ON (높이 ±5cm, 관절 ±0.1rad) | 완주 + 실측 반복 | 충격 477→**194N**, 오차 0.88→**0.094m** |

승급 판정은 보상 점수가 아니라 **실측 물리량**(자이로 적산 회전각 등)으로 합니다 — 보상 스케일을 바꿔도 판정 기준이 흔들리지 않게. 감독 스크립트가 60초마다 로그를 확인해 기준을 3회 연속 충족하면 자동으로 다음 스테이지를 시작합니다.

**왜 이 순서인가**: 점프 없이는 회전할 시간이 없으므로 도약이 먼저, 180°는 "등으로 떨어져도 되는" 안전한 중간 목표, 360°에서 발 착지를 강제, 그다음에야 강건화·정밀화 — 각 단계가 이전 단계의 스킬을 전제로 하고, 도약 보상은 전 단계에 유지되어 승급이 "수업 종료"가 아닌 "수업 추가"가 되게 했습니다.

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
