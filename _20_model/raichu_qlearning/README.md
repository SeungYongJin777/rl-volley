# raichu_qlearning

Champion-oriented Q-learning line copied from the base `qlearning` package and then upgraded for stronger play.

## Base

- Base algorithm: Q-learning
- Base implementation: `qlearning`
- Default flagship policy: `million_volts`

## Changes

- Replaces the original wide state vector with a compressed rally-focused state
- Adds legal action masking plus an internal `idle` action to reduce invalid exploration
- Adds heuristic priors for unseen states so early exploration is less random
- Uses visit-count-based alpha decay so late training is more stable
- Uses a stronger win-oriented reward with point, match, positioning, and spike readiness shaping
- Uses a separate model package and output directory so Raichu experiments do not overwrite `qlearning`

## Training

Start fresh:

```bash
.venv/bin/python cli.py --mode train --train_algorithm raichu_qlearning --train_side 1p --train_policy million_volts --train_rewrite True --train_opponent rule --num_episode 50000 --target_score 5
```

Continue training:

```bash
.venv/bin/python cli.py --mode train --train_algorithm raichu_qlearning --train_side 2p --train_policy million_volts --train_opponent self --num_episode 50000 --target_score 5
```

## Notes

- Use a new seed when continuing training with `train_rewrite=False`
- Promote only the best gauntlet result to `million_volts`
- `epsilon_decay` is slowed for overnight cycling so exploration does not collapse in the first few thousand episodes
- Checkpoints are saved at `5k, 10k, 15k, 20k, 25k, 30k`

## Overnight Automation

Run one foreground cycle loop:

```bash
.venv/bin/python _30_src/automate_raichu_train.py
```

Start detached in `screen`:

```bash
.venv/bin/python _30_src/automate_raichu_train.py --launch-screen
```

Default behavior:

- Cycles `1p rule -> 2p rule -> 1p self -> 2p self`
- Uses `30000` episodes per stage
- Uses a new seed for every stage
- Bootstraps with `train_rewrite=True` only if `million_volts.pt` does not exist
- Writes logs under `outputs/automation_logs/`
