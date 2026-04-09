# rl-volley CLI 예문

## Play

- rule vs <algorithm_name>:<policy_name>

```bash
python3 cli.py --mode play --1p rule --2p sarsa:100000 --target_score 15
```

- human vs <algorithm_name>:<policy_name>

```bash
python3 cli.py --mode play --1p human --2p dqn:1 --target_score 15
```

- <algorithm_name>:<policy_name> vs <algorithm_name>:<policy_name>

```bash
python3 cli.py --mode play --1p qlearning:100000 --2p sarsa:100000 --target_score 15
```


## Train

- 알고리듬: qlearning / 학습 위치: 1p / 학습 정책명: test / 새로 학습하기 여부: True / 대결 대상: rule / 학습 에피소드 수: 1000

```bash
python3 cli.py --mode train --train_algorithm qlearning --train_side 1p --train_policy test --train_rewrite True --train_opponent rule --num_episode 1000
```

- 알고리듬: sarsa / 학습 위치: 1p / 학습 정책명: test / 새로 학습하기 여부: True / 대결 대상: self (실시간 미러링) / 학습 에피소드 수: 1000

```bash
python3 cli.py --mode train --train_algorithm sarsa --train_side 1p --train_policy test --train_rewrite True --train_opponent self --num_episode 1000
```

- 알고리듬: dqn / 학습 위치: 2p / 학습 정책명: test / 새로 학습하기 여부: True / 대결 대상: <algorithm_name>:<policy_name> / 학습 에피소드 수: 1000

```bash
python3 cli.py --mode train --train_algorithm dqn --train_side 2p --train_policy test --train_rewrite True --train_opponent qlearning:100000 --num_episode 1000
```

