
# PPO 기반 토너먼트 대응형 Pikachu Volleyball 학습 구조 개편 정리

## 목적

현재 구조는 **학습 모델 1개 vs 고정 상대 1개** 형태로 학습하는 구조이다.  
하지만 실제 목표가 **토너먼트 형식에서 다양한 모델을 상대로도 안정적으로 이기는 모델**을 만드는 것이라면, 단일 고정 상대만 두고 학습하는 방식은 한계가 있다.

따라서 가장 적절한 방향은 다음과 같다.

- **모델 자체는 PPO를 유지**
- **학습 방식은 opponent pool 기반 multi-opponent training으로 변경**
- **필요 시 snapshot self-play를 추가**
- **평가 기준도 단일 상대 기준이 아니라 다중 상대 평균 성능 기준으로 변경**

즉, 이번 작업의 본질은 **새로운 네트워크를 만드는 것**보다  
**학습 상대를 어떻게 구성하고 순환시키느냐를 바꾸는 것**에 있다.

---

# 1. 최종 방향

현재 구조에서는 새 알고리즘을 완전히 갈아엎기보다,

## 기존 PPO 구조를 유지하면서
## 상대 선택 로직을 “고정 opponent 1개”에서 “opponent pool”로 바꾸는 방향

이 가장 자연스럽다.

즉 핵심은 다음과 같다.

- 모델 자체: **PPO**
- 학습 방식: **다중 상대 기반 학습**
- 목적: **토너먼트에서 어떤 상대를 만나도 대응 가능하도록 일반화**

---

# 2. 현재 train 코드에서 보이는 핵심 구조

현재 `train.py`의 흐름은 다음과 같다.

## 현재 학습 흐름
1. 환경 생성
2. 학습 모델 하나 로드
3. 상대 모델 하나 로드
4. 에피소드마다 `env.set(player1=..., player2=...)`
5. `model_train.get_transition(...)`
6. `model_train.update(transition)`
7. 끝나면 저장

즉 지금 구조는 사실상:

## “학습 모델 1개 vs 고정 상대 1개”

형태이다.

토너먼트 대응형으로 바꾸려면 여기서 제일 먼저 바뀌어야 하는 것은:

## `model_opponent`를 단일 객체가 아니라 “선택 가능한 상대 집합”으로 바꾸는 것

이다.

---

# 3. 가장 추천하는 설계 방향

## 핵심 아이디어
`train.py`에서는 매 에피소드마다 상대를 하나 뽑고,  
그 상대와 붙여서 학습하게 만든다.

즉 현재 구조:

```python
model_opponent = load_model(...)
```

이 부분을 아래 구조로 바꾸는 것이 핵심이다.

```python
opponent_pool = create_opponent_pool(...)
model_opponent = opponent_pool.sample()
```

---

# 4. 어디를 어떻게 바꾸면 되는지

---

## 4-1. `train.py`에서 바꿔야 할 것

현재는 학습 시작 전에 상대를 한 번만 로드한다.

```python
model_train = load_model(conf, player=conf.train_side)
model_opponent = load_model(conf, player='1p' if conf.train_side == '2p' else '2p')
```

이 부분을 아래처럼 바꾸는 것이 제일 중요하다.

## 바꿀 방향
- `model_train`은 그대로 1개
- `model_opponent`는 에피소드마다 샘플링
- 상대 후보는 rule, random, old checkpoint, external PPO 등 여러 개

### 추천 구조
```python
model_train = load_model(conf, player=conf.train_side)
opponent_pool = create_opponent_pool(conf, model_train)
```

그리고 에피소드 루프 안에서:

```python
for epi_idx in tqdm(...):
    model_opponent = opponent_pool.sample_opponent(epi_idx)
```

이렇게 바꾸는 식이 가장 자연스럽다.

---

## 4-2. 새로 필요한 개념: `OpponentPool`

이 기능은 꼭 들어가는 것이 좋다.

### 역할
- 상대 후보 목록 관리
- 상대 샘플링
- 과거 checkpoint 추가
- 외부 모델 추가
- rule/random heuristic 포함
- 에피소드별 또는 주기별 상대 교체

이 기능을 어디에 둘지가 중요한데, 현재 파일 구조를 보면 가장 자연스러운 위치는 다음 둘 중 하나이다.

## 추천 위치
### **`_06_algorithm.py` 안에 넣기**
또는
### **새 파일 `_07_opponent_pool.py` 추가**

하지만 지금 구조를 크게 안 흔들려면 우선 `_06_algorithm.py` 안에 클래스로 넣는 것이 가장 무난하다.

---

# 5. 파일별 역할 추천

---

## 5-1. `_00_model.py`

여기는 **모델 생성과 조립 담당**으로 두는 것이 좋다.

현재도 `create_model(...)` 역할을 하는 쪽일 가능성이 높다.  
여기서는 크게 바꾸기보다, PPO 모델 생성 시 **추가 옵션**만 받게 하면 된다.

### 여기서 추가하면 좋은 것
- `policy_name_for_play`
- `is_train`
- `checkpoint_path`
- `fixed_for_opponent`
- `eval_mode`

즉, 같은 PPO라도
- 학습용 모델
- 상대용 frozen 모델

을 구분해서 만들 수 있도록 하면 좋다.

---

## 5-2. `_01_params.py`

여기가 이번 개조의 핵심 설정 파일이 될 가능성이 크다.

여기에 아래 파라미터들을 추가하는 것을 추천한다.

## 꼭 넣을 파라미터
```python
# Opponent Pool
use_opponent_pool = True
opponent_sampling_mode = "per_episode"   # per_episode / fixed_interval
opponent_swap_interval = 1               # 1이면 매 episode마다 교체

# Opponent Types
use_rule_opponent = True
use_random_opponent = True
use_external_opponents = True
use_snapshot_opponents = True

# Sampling Ratios
rule_ratio = 0.25
random_ratio = 0.10
snapshot_ratio = 0.40
external_ratio = 0.25

# Snapshot
save_snapshot_interval = 100
max_snapshot_size = 10
snapshot_dir = "./checkpoints_snapshots"

# Evaluation
eval_interval = 50
eval_num_episode_per_opponent = 20

# Robustness
use_hard_negative_sampling = False
hard_negative_ratio = 0.3
```

### 왜 중요한가
이 파라미터들을 분리해두면 나중에 실험할 때
- self-play를 켜고 끄기 쉽고
- 외부 모델 비율을 바꾸기 쉽고
- 토너먼트용 튜닝이 쉬워진다

---

## 5-3. `_02_network.py`

여기는 **PPO 네트워크 자체**를 담당하므로 크게 복잡하게 안 건드리는 것이 좋다.

### 추천
- 지금 PPO 네트워크 유지
- 너무 크지 않은 MLP
- 입력: 상태
- 출력: policy logits + value

즉 토너먼트 대응력은 주로 여기서 나오는 것이 아니라  
**상대 분포를 어떻게 구성하느냐**에서 나온다.

그래서 여기는 구조를 갈아엎지 말고 안정적으로 두는 것이 좋다.

---

## 5-4. `_03_state_design.py`

토너먼트형이면 여기 중요도가 올라간다.

왜냐하면 다양한 상대를 만나려면 state가 상대의 스타일 차이를 어느 정도 반영해야 하기 때문이다.

## 여기서 추가하면 좋은 정보
현재 state에 아래가 없다면 검토할 만하다.

- 상대 최근 위치 변화
- 상대 최근 행동
- 공의 속도 방향
- 공이 내 코트로 오는지/상대 코트로 가는지
- 점수 차
- 현재 서브 상황
- 최근 몇 step 동안 상대 점프 빈도

### 핵심 포인트
상대를 직접 “이 상대는 공격형이다”라고 분류할 필요는 없지만,
**상대 행동 패턴이 state에 드러나게 하는 것**이 중요하다.

예를 들어:
- 직전 상대 action
- 최근 3-step 상대 이동 방향
- 최근 상대 점프 여부

이런 것만 있어도 범용 대응력이 좋아질 수 있다.

---

## 5-5. `_04_action_space_design.py`

여기는 행동 공간을 너무 복잡하게 만들 필요는 없다.

### 추천 방향
- 기존 action set 유지
- 다만 “의미 없는 행동”이 너무 많으면 정리
- 수비/공격의 기본 행동 조합이 충분한지만 확인

즉 여기는 “새 기능 추가”보다
**action space가 PPO가 배우기에 너무 어려운 형태는 아닌지 확인하는 정도**가 좋다.

---

## 5-6. `_05_reward_design.py`

여기도 중요하지만, 토너먼트용일수록 오히려 **단순한 것이 낫다.**

## 추천 reward
- 득점: `+1`
- 실점: `-1`
- 그 외: `0`

필요하면 아주 약한 shaping만 추가한다.

예:
- 공을 넘기면 `+0.02`
- 공을 받으면 `+0.01`
- 비효율적 반복 점프 패널티 `-0.005`

### 주의
reward를 과하게 꾸미면  
“상대를 이기는 법”보다 “reward를 잘 따먹는 움직임”을 학습할 수 있다.

토너먼트에서는 결국 **승률**이 가장 중요하므로,
reward는 최대한 승패 중심이 좋다.

---

## 5-7. `_06_algorithm.py`

여기가 이번 개조의 중심이다.

현재 파일명이 `_06_algorithm.py`인 걸 보면
PPO update나 rollout 관련 로직이 여기에 있을 가능성이 크다.

여기에 아래 기능들을 넣는 것을 추천한다.

---

# 6. `_06_algorithm.py`에 추가하면 좋은 기능

## 6-1. `OpponentPool` 클래스

가장 중요하다.

### 역할
- 상대 후보 저장
- 비율에 따라 샘플링
- snapshot 등록
- 외부 모델 등록
- 최근 패배율 기반 샘플링 확장 가능

### 대략적인 구조
```python
class OpponentPool:
    def __init__(self, conf, train_side):
        self.conf = conf
        self.train_side = train_side
        self.rule_opponents = []
        self.random_opponents = []
        self.snapshot_opponents = []
        self.external_opponents = []

    def add_rule_opponent(self, opponent):
        ...

    def add_snapshot_opponent(self, opponent):
        ...

    def add_external_opponent(self, opponent):
        ...

    def sample_opponent(self):
        ...
```

---

## 6-2. snapshot 저장 함수

학습 모델을 주기적으로 저장해서 상대 pool에 넣어야 한다.

### 왜 필요한가
- 과거의 나와 싸워야 특정 스타일 과적응을 막을 수 있음
- self-play의 핵심

### train loop에서 쓰는 방식
예:
```python
if epi_idx % conf.save_snapshot_interval == 0:
    snapshot_path = model_train.save_snapshot(epi_idx)
    opponent_pool.register_snapshot(snapshot_path)
```

---

## 6-3. 다중 상대 평가 함수

best model을 그냥 “학습 reward 제일 높음”으로 뽑지 말고,  
여러 상대와 붙여 본 평균 승률로 판단해야 한다.

### 추천 함수
```python
def evaluate_against_pool(model_train, opponent_pool, env, conf):
    ...
```

### 평가 결과로 저장할 것
- 상대별 승률
- 전체 평균 승률
- 최악 상대 승률
- score differential 평균

---

## 6-4. hard negative sampling

이건 나중 확장용이지만 굉장히 좋다.

### 개념
자주 지는 상대를 더 자주 만나게 하는 것

예:
- 특정 공격형 모델에게 약함
- 그러면 그 상대 샘플링 확률을 높임

초기엔 없어도 되지만, 나중엔 효과가 좋다.

---

# 7. train.py 기준으로 실제 수정 포인트

---

## 7-1. 현재 코드의 가장 중요한 수정 부분

현재:

```python
model_train = load_model(conf, player=conf.train_side)
model_opponent = load_model(conf, player='1p' if conf.train_side == '2p' else '2p')
```

이 부분을 이렇게 바꾸는 방향이 좋다.

## 수정 방향
```python
model_train = load_model(conf, player=conf.train_side)
opponent_pool = create_opponent_pool(conf, model_train)
```

그리고 에피소드 안에서:

```python
for epi_idx in tqdm(range(conf.num_episode), desc="Training Progress"):
    model_opponent = opponent_pool.sample_opponent()
```

---

## 7-2. snapshot 추가

에피소드 끝날 때 또는 일정 간격마다:

```python
if epi_idx % conf.save_snapshot_interval == 0 and epi_idx > 0:
    snapshot_path = model_train.save_snapshot(epi_idx)
    opponent_pool.add_snapshot(snapshot_path)
```

---

## 7-3. eval 추가

일정 간격마다:

```python
if epi_idx % conf.eval_interval == 0 and epi_idx > 0:
    evaluate_model_against_pool(model_train, opponent_pool, conf)
```

---

# 8. 추천 train.py 개조 흐름

## 개조 후 흐름
1. env 생성
2. model_train 생성
3. opponent_pool 생성
4. 에피소드 시작
5. 상대 샘플링
6. `env.set(player1, player2)`
7. episode 학습
8. 일정 주기마다 snapshot 저장
9. 일정 주기마다 pool 평가
10. 최종 모델 저장

---

# 9. 가장 추천하는 상대 구성

토너먼트용이면 pool 구성 자체가 중요하다.

## 추천 opponent pool 구성

### A. 고정 상대
- rule-based baseline 1개
- random baseline 1개

### B. heuristic 상대
- 수비형 heuristic
- 공격형 heuristic

### C. self-play 상대
- 내 과거 checkpoint 5~10개

### D. 외부 상대
- 다른 팀/사람 모델 checkpoint

---

# 10. 가장 추천하는 샘플링 비율

처음엔 너무 복잡하게 하지 말고 다음 정도로 시작한다.

- rule: 30%
- random/heuristic: 20%
- past snapshots: 30%
- external models: 20%

조금 더 토너먼트 후반 느낌으로 가려면:

- snapshots: 40%
- external: 30%
- rule: 20%
- random/heuristic: 10%

---

# 11. 지금 단계에서 가장 좋은 구현 우선순위

## 1단계
### `train.py`에서 opponent를 단일 객체 → pool 샘플링 구조로 변경

이게 1순위이다.

---

## 2단계
### `_06_algorithm.py`에 `OpponentPool` 추가

이게 2순위이다.

---

## 3단계
### `_01_params.py`에 pool 관련 설정 추가

이게 3순위이다.

---

## 4단계
### snapshot 저장 및 로드 기능 추가

이게 4순위이다.

---

## 5단계
### evaluation을 상대별 평균 승률 기준으로 변경

이게 5순위이다.

---

# 12. 코덱스에게 넘기기 좋은 형태로 요약

아래처럼 전달하면 된다.

---

## 작업 목표
기존 PPO 기반 학습 구조를 유지하되,  
고정 opponent 1개와만 학습하는 방식에서 벗어나  
**opponent pool 기반 multi-opponent training** 구조로 변경한다.

## 핵심 변경 사항
1. `train.py`에서 `model_opponent`를 단일 모델로 고정하지 말고, `OpponentPool`에서 episode마다 샘플링하도록 수정
2. `_06_algorithm.py`에 `OpponentPool` 클래스를 추가해 rule/random/snapshot/external opponent를 관리하도록 구현
3. `_01_params.py`에 opponent pool 관련 파라미터 추가
4. 일정 episode마다 `model_train`의 snapshot을 저장하고 pool에 추가
5. 일정 episode마다 다중 상대 평가를 수행하고 평균 승률/최악 승률 기준으로 best model을 기록
6. `_03_state_design.py`에서 상대 최근 행동/상대 위치 변화 등 상대 패턴을 반영할 최소 상태 확장 검토
7. `_05_reward_design.py`는 승패 중심 reward를 유지하고 shaping은 최소화

## 구현 우선순위
1. opponent pool 샘플링 구조
2. snapshot self-play
3. multi-opponent evaluation
4. 상태 확장
5. hard negative sampling

---

# 13. 아주 짧은 최종 결론

## 네 구조에서는
**모델 자체는 PPO로 유지하고, `train.py`와 `_06_algorithm.py` 중심으로 “상대 선택 구조”를 고정 1명에서 opponent pool 방식으로 바꾸는 것이 가장 좋다.**

즉, 이번 작업의 본질은 **새 신경망을 만드는 것**보다  
**학습 상대를 어떻게 구성하고 순환시키느냐를 바꾸는 것**이다.

---

# 14. 현재 train.py 코드 참고용

```python
from tqdm import tqdm

# Import Required Internal Modules
import _00_environment
import _20_model
from _30_src.play import load_model


def run(conf):
    \"\"\"====================================================================================================
    ## Create Required Instances
    ====================================================================================================\"\"\"
    # - Create Envionment Instance
    env = create_environment_instance(conf)

    \"\"\"====================================================================================================
    ## Run a number of Episodes for Training
    ====================================================================================================\"\"\"
    # - Load Models for Training and Opponent Players
    model_train = load_model(conf, player=conf.train_side)
    model_opponent = load_model(
        conf, player='1p' if conf.train_side == '2p' else '2p')

    # - Run a number of Episodes for Training
    for epi_idx in tqdm(range(conf.num_episode), desc="Training Progress"):
        # - Set the Environment
        if conf.train_side == '1p':
            env.set(player1=model_train, player2=model_opponent,
                    random_serve=conf.random_serve, return_state=False)
        else:
            env.set(player1=model_opponent, player2=model_train,
                    random_serve=conf.random_serve, return_state=False)

        # - Get Initial State
        state_mat = env.get_state(player=conf.train_side)

        # - Run an Episode
        while True:
            # - Get Transition by Action Selection and Environment Run
            transition, state_next_mat = model_train.get_transition(
                env, state_mat)

            # - Update Policy by Transition
            model_train.update(transition)
            env = model_train.env

            # - Update State
            state_mat = state_next_mat

            # - Check Terminate Condition
            done = transition[-2]
            if done:
                break

    \"\"\"====================================================================================================
    ## Save Trained Policy at the End of Episode
    ====================================================================================================\"\"\"
    model_train.save()


def create_environment_instance(conf):
    \"\"\"====================================================================================================
    ## Creation of Environment Instance
    ====================================================================================================\"\"\"
    # - Load Configuration
    RENDER_MODE = "log"
    TARGET_SCORE = conf.target_score_train
    SEED = conf.seed

    # - Create Envionment Instance
    env = _00_environment.Env(
        render_mode=RENDER_MODE,
        target_score=TARGET_SCORE,
        seed=SEED,
    )

    # - Return Environment Instance
    return env


def load_model(conf, player):
    \"\"\"====================================================================================================
    ## Loading Policy for Each Player
    ====================================================================================================\"\"\"
    # - Check Algorithm and Policy Name for Each Player
    ALGORITHM = conf.algorithm_1p if player == '1p' else conf.algorithm_2p
    POLICY_NAME = conf.policy_1p if player == '1p' else conf.policy_2p

    # - Load Selected Policy for Each Player
    if ALGORITHM == 'human':
        model = 'HUMAN'

    elif ALGORITHM == 'rule':
        model = 'RULE'

    else:
        model = _20_model.create_model(
            conf,
            algorithm_name=ALGORITHM,
            policy_name_for_play=POLICY_NAME,
        )

    # - Return Loaded Model for Each Player
    return model


if __name__ == "__main__":
    pass
```

---

# 15. 현재 파일 구조 참고용

```text
__pycache__/
_00_model.py
_01_params.py
_02_network.py
_03_state_design.py
_04_action_space_design.py
_05_reward_design.py
_06_algorithm.py
__init__.py
```
