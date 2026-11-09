# Experimental configurations

This page lists every setting of the experiments in the paper, the commands that run them and what each run
produces. The extended result tables and the hardware the runs used are in the appendix of the paper.

- [Common settings](#common-settings)
- [(P1) T-maze](#p1-t-maze)
- [Slack in place of memory](#slack-in-place-of-memory)
- [(P2) RockSample](#p2-rocksample)
- [(P2) Polar maze](#p2-polar-maze)
- [CGCP with SARSOP](#cgcp-with-sarsop)
- [Tables and figures](#tables-and-figures)
- [Run times](#run-times)

## Common settings

- **Values.** Every value is computed on the product model by the occupancy recursion. Nothing is
  sampled, so a run with a given seed gives the same numbers on every machine up to floating-point round-off.
- **Seeds.** Every method runs on seeds 0 to 49.
- **Sweeps.** A sweep of CPI updates the stages backwards in time, from $`t=T`$ to $`t=1`$. At stage $`t`$ every row
  with positive occupancy moves toward the stage rule $`g_t`$,

  ```math
  \pi_t \;\leftarrow\; (1-\alpha)\,\pi_t + \alpha\, g_t .
  ```

  With $`\alpha=1`$ we use the lazy rule, which keeps the rows that are already greedy.
- **Stopping.** A run stops when no row changes, when the objective $`J`$ gains at most $`10^{-9}\max\{1,\lvert J\rvert\}`$
  in two consecutive sweeps, or after 60 sweeps.
- **The (P2) schedule.** CPI runs satisfaction sweeps until $`J_\varphi\ge c`$ and then the joint sweeps of
  Proposition 2.

## (P1) T-maze

```
G0  D0  X  D1  G1      the bar: each side passes through a door to a goal
        J              the junction
        K              two keys on stem tiles, at positions drawn from the seed
        K
        C              the start tile, which shows the cue at t = 1 only
       [m]             the start margin, m tiles behind the start
```

- **Stem.** $`L`$ tiles lead from the start tile to the junction. Behind the start tile there are $`m`$ further tiles.
- **Keys and doors.** Key $`i`$ opens door $`i`$. The agent can pick a key only when it holds none, and it never drops
  a key. Opening a door without its key ends the episode, and so does moving into a closed door.
- **Actions.** Forward, backward, left, right, pick and open. Actions that would leave the agent in place are illegal,
  so the agent cannot wait. The cheapest delay is one step back and one step forward, which needs $`m\ge1`$.
- **Observation.** The position, a token and two bits that tell whether the agent holds a key and whether a door is
  open. The token is the cue at $`t=1`$, a key on a tile that holds one, and otherwise the event of the last action. The
  bits never say which key or which door.
- **Specification.** The labels are $`k`$ (a key was picked), $`d`$ (a door was opened) and $`g_c`$ (the cued goal was
  reached), and

  ```math
  \varphi=\mathsf F\big(k\wedge\mathsf F(d\wedge\mathsf F\,g_c)\big).
  ```

  The cue reaches the automaton only at the goal.
- **Horizon.** The shortest accepting route takes $`L+5`$ steps, and the horizon is $`T=L+5+W`$ for a slack $`W`$.

| Setting | Values |
|---|---|
| Corridor length $`L`$ | 3, 5, 10, 20 |
| Slack $`W`$ | 0, 1, 2, 3, 4, 5, 6 |
| Start margin $`m`$ | 0, 1 |
| Memory $`\lvert\mathcal Z\rvert`$ | 1, 2 |
| Step size $`\alpha`$ | 0.9, and 1 with the lazy rule |
| Start | Uniform legal actions, with the copy kernel $`\mathbf 1\{z'=z\}`$ on the memory |
| Stage rule | Greedy on the satisfaction value. Ties go to the larger reward value ($`+1`$ at the cued goal, $`-1`$ at the other), and the remaining ties are broken at random by the seed. |
| Seeds | 50 per layout. The seed also places the keys. |
| Runs | 56 layouts × 2 memory sizes × 2 step sizes × 50 seeds = 11,200 |

```bash
python suite.py --result remark7 --workers 14     # writes runs_main/tmaze_main and runs_paper/tmaze_slack
python plot_tmaze_curves.py                        # the learning curves with every seed
```

With $`\lvert\mathcal Z\rvert=2`$ all 5,600 runs reach $`J_\varphi=1`$ at both step sizes. With $`\alpha=0.9`$ the median
number of sweeps grows from 12 at $`L=3`$ to 14 at $`L=20`$, and with lazy updates it is 3 at every length. Without
memory only 165 of the 5,600 runs reach $`J_\varphi=1`$. All of them have $`\alpha=0.9`$, $`m=1`$ and $`W\ge4`$.

## Slack in place of memory

With $`m\ge1`$ and $`W\ge2`$, a policy without memory can carry the cue in its position. It steps back at $`t=1`$ for one
cue value and forward for the other. The two cue values are then on different tiles at every later step, and a
time-varying policy can read the cue from the position. `tmaze_route.py` builds this route policy and evaluates it
exactly. This experiment tests whether CPI finds such policies on its own.

| Setting | Values |
|---|---|
| Memory $`\lvert\mathcal Z\rvert`$ | 1 |
| Start margin $`m`$ | 1 |
| Corridor length $`L`$ | 3, 5, 10, 20 |
| Slack $`W`$ | 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16 |
| Step size $`\alpha`$ | 0.9, and 1 with the lazy rule |
| Start, stage rule and ties | As in the T-maze above |
| Runs | 4 lengths × 11 slacks × 2 step sizes × 50 seeds = 4,400, and the route policy on all 2,200 layouts |

```bash
python suite.py --result remark7 --workers 14     # the same command as above
python plot_tmaze_slack.py --root runs_paper --out reports_paper
```

The route policy reaches $`J_\varphi=1`$ on every layout with $`W\ge2`$ and on none with $`W<2`$. CPI with $`\alpha=0.9`$
first finds such a policy at $`W=4`$, and there only for $`L=3`$. On longer corridors it finds one far less often, and
lazy updates never find one.

<p align="center">
  <img src="reports_paper/figA_tmaze_slack.png" alt="Share of memoryless runs that reach satisfaction one, against the slack" width="55%"><br>
  <sub>(P1) without memory and with start margin 1. The share of runs that reach J<sub>φ</sub> = 1, against the slack.</sub>
</p>

## (P2) RockSample

| | Grid | Rocks | Start | Horizon $`T`$ | Ceiling $`1-2^{-K}`$ | Thresholds $`c`$ | Product nodes |
|---|---|---|---|---|---|---|---|
| RS(5,3) | 5×5 | (1,1) (3,3) (4,4) | (1,1) | 32 | 0.875 | 0.55, 0.8, 0.86 | 1,751 |
| RS(5,5) | 5×5 | (2,3) (4,4) (4,2) (3,5) (5,3) | (1,1) | 40 | 0.969 | 0.55, 0.9, 0.96 | 7,757 |

- **Rocks.** Each rock is good with probability $`1/2`$, independently of the others. Cells are numbered from 1, with
  $`x`$ to the east and $`y`$ to the north.
- **Actions.** Sample, north, east, south, west, and one check per rock. A check of a rock at distance $`d`$ returns its
  quality with probability

  ```math
  \tfrac12\big(1+2^{-d/20}\big).
  ```

- **Rewards.** Sampling a good rock pays 10 and makes the rock bad. Moving east from the last column exits the grid, pays
  10 and ends the episode. Sampling a bad rock pays nothing, so only the specification forbids it. The presets default
  to a penalty of $`-10`$, so pass `--bad-rock-penalty 0` (or `bad_rock_penalty=0.0`) to match the paper.
- **Discount.** The reward is discounted by $`\gamma=0.95`$. Without the discount, checking a rock costs nothing. The
  agent could then check every rock and sample only good ones, and reward and satisfaction would no longer conflict.
- **Observation and labels.** The observation is the position, the reading of the last check and the outcome of the last
  action. The labels good, bad and exit come from that outcome, and the agent never observes the quality of a rock
  directly. The specification is

  ```math
  \varphi=\mathsf F(\mathrm{good}\wedge\mathsf F\,\mathrm{exit})\;\wedge\;(\neg\mathrm{bad}\;\mathsf U\;\mathrm{exit}).
  ```

| Method | Settings |
|---|---|
| CPI, $`\lvert\mathcal Z\rvert=2`$ and $`\lvert\mathcal Z\rvert=4`$ | Rows drawn from a Dirichlet distribution with all parameters equal to one, seeded by the seed and the row. Ties broken by action index, and $`\alpha=0.9`$. |
| Trial-and-Error | The same start and schedule as CPI. Each stage moves toward the reward-greedy rule with the largest step, at most $`\alpha`$, that keeps $`J_\varphi\ge c`$. |
| CPI, best of $`k`$ | The expected best of $`k`$ CPI runs with $`\lvert\mathcal Z\rvert=2`$, drawn without replacement from the 50 runs. $`k`$ is the median number of sweeps of LBS divided by that of CPI, rounded, which gives 17 to 20. |
| LBS | A (P1) ascent from the initial policy gives $`\tilde\pi`$ and $`\tilde\kappa=J_\varphi(\tilde\pi)-c`$. The first bracket is $`[0,R/\tilde\kappa]`$, followed by 14 midpoint steps, so LBS prices 16 multipliers. For each multiplier, Best runs two ascents on $`L_\lambda`$ with $`\alpha=0.9`$, one from $`\tilde\pi`$ and one from the initial policy. It also considers the policies already on the upper hull of everything evaluated, whose values are stored. |
| LBS + CPI | The joint sweeps of CPI on the output of LBS, with the latch bit as a second memory component. The polished policy has four memory states. |

```bash
python suite.py --result thm6 --workers 14        # writes runs_paper/rock (and a few arms the paper does not report)
python paper.py --exp rock --include rocksample5 --workers 14    # only the five-rock runs
python plot_p2.py --no-npg --cloud-ours-only       # writes reports_p2/
python cheap_bounds.py rocksample5_tight 0.55 0.9 0.96           # the simple upper bounds below
```

## (P2) Polar maze

```
G0 ── J ── G1      at the junction J the agent turns left or right and reaches a goal
      │
      S            K survey tiles, spread evenly over the stem
      ⋮
      S
      C            the cue tile, which shows the correct goal at t = 1 only
```

- **Stem.** The agent walks $`2K+1`$ steps from the cue tile to the junction. It moves forward one tile per step and
  cannot wait or go back. On a survey tile it can scan, which scans the tile and moves forward.
- **Horizon and discount.** $`T=2K+2`$, and there is no discount.
- **Reward and specification.** The reward is one if the agent scanned all $`K`$ tiles. The specification
  $`\varphi=\mathsf F\,g_c`$ holds if the agent reaches the cued goal after at most $`s`$ scans. No observation and no label
  depends on the number of scans.
- **Methods.** CPI with $`\lvert\mathcal Z\rvert=2`$ and $`\lvert\mathcal Z\rvert=4`$, and LBS + CPI with
  $`\lvert\mathcal Z\rvert=2`$. Every run starts from uniform legal actions, with a fair coin written into memory at
  $`t=1`$ and copied afterwards. Ties are broken at random by the seed, and $`\alpha=0.9`$.
- **The value $`v_2(c)`$.** It has a closed form for $`s=0`$ and for $`s=K-1`$. For the other budgets we report the best
  value that CPI with $`\lvert\mathcal Z\rvert=2`$ finds from 600 starts, which is a lower bound on $`v_2(c)`$.

| $`K`$ | Budget $`s`$ | Thresholds $`c`$ | Horizon $`T`$ | Preset |
|---|---|---|---|---|
| 2 | 0 | 0.6, 0.75, 0.9 | 6 | `tgap_K2_s0` |
| 3 | 0 | 0.75 | 8 | `tgap_K3_s0` |
| 4 | 0 | 0.6, 0.75, 0.9 | 10 | `tgap_K4_s0` |
| 4 | 1, 2, 3 | 0.75 | 10 | `tgap_K4_s1`, `tgap_K4_s2`, `tgap_K4_s3` |
| 6 | 0 | 0.75 | 14 | `tgap_K6_s0` |

```bash
python suite.py --result prop5 --workers 14       # writes runs_paper/tgap
python tgap_search.py --n 300 --root runs_paper    # the best value found where v2 has no closed form
```

## CGCP with SARSOP

CGCP (Walraven and Spaan, 2018) is the belief-space reference. It has no memory budget.

- **Master problem.** A linear program over the policies found so far, with one constraint for $`J_\varphi\ge c`$ and one
  for the weights summing to one. Its dual gives $`\lambda`$.
- **Pricing.** Each pricing step maximizes $`L_\lambda`$ with SARSOP on the product POMDP unrolled over the horizon. The
  unrolled model has a start state, a state $`(t,\text{node})`$ for every node reachable at step $`t`$, and an absorbing
  end state. SARSOP needs a discount $`\gamma_s<1`$, so we use $`\gamma_s=0.99`$ and rescale the rewards so that its
  discounted value equals the finite-horizon value exactly. With $`t=0,\dots,T-1`$ in code indexing and $`F`$ the
  accepting automaton states,

  ```math
  \tilde r\big((t,s),a\big)=\frac{\gamma^{t}\,r(s,a)}{\gamma_s^{\,t+1}},
  \qquad
  \tilde r\big((T,s)\big)=\frac{\lambda\,\mathbf 1\{q(s)\in F\}}{\gamma_s^{\,T+1}} .
  ```

- **Evaluation.** Every new policy is evaluated exactly on its belief tree.
- **Stopping.** The method stops when no policy has a positive reduced cost, or when two consecutive policies leave the
  master value unchanged. A final, longer SARSOP call at the last multiplier tightens the upper bound.
- **Upper bound.** SARSOP also returns an upper bound $`U(\lambda)`$ on $`\max L_\lambda`$ over all belief-space policies.
  Over the multipliers tried,

  ```math
  \bar v_2(c)\;\le\;\min_{\lambda}\,\big(U(\lambda)-\lambda c\big),
  ```

  which bounds every method in the comparison.

| | RS(5,3) | RS(5,5) | Polar maze |
|---|---|---|---|
| Unrolled states | 46,988 | 263,608 | small |
| Limit per SARSOP call | 120 s | 30 min | 30 s |
| Limit of the final call | 600 s | 2 h | 600 s |
| Precision | $`10^{-3}`$ | $`10^{-3}`$ | $`10^{-3}`$ |

### Building SARSOP

We use SARSOP at commit `d914110` of [AdaCompNUS/sarsop](https://github.com/AdaCompNUS/sarsop). With gcc 10 or later
the build needs `-fcommon`.

```bash
git clone https://github.com/AdaCompNUS/sarsop.git
git -C sarsop checkout d914110
sed -i '/^CFLAGS /s/\$(CYGWIN_CFLAGS)/-fcommon $(CYGWIN_CFLAGS)/' sarsop/src/Makefile
make -C sarsop/src -j"$(nproc)"
export SARSOP="$PWD/sarsop/src/pomdpsol"
```

The model files spell out every action, and `cgcp_sarsop.py` loads them with `pomdpsol --fast`. The default parser
takes hours on the five-rock model. On five rocks each SARSOP call needs several GB of memory and writes several GB of
model and policy files, so point `--workdir` at a local scratch disk. Running the thresholds one at a time lowers the
peak memory.

### Running CGCP

```bash
# three rocks, one threshold after another
python cgcp_sarsop.py --task rocksample3_tight --c 0.55 0.8 0.86 --root runs_paper

# the polar maze, one call per setting
python cgcp_sarsop.py --task tgap_K2_s0 --c 0.6 0.75 0.9 --timeout 30 --root runs_paper

# five rocks: the two shared columns first, then the thresholds (in parallel if memory allows)
W=/path/to/scratch            # SARSOP's model and policy files, several GB
python cgcp_sarsop.py --task rocksample5_tight --c 0.55 0.9 0.96 --only reward       --timeout 1800 --workdir $W --root runs_paper &
python cgcp_sarsop.py --task rocksample5_tight --c 0.55 0.9 0.96 --only satisfaction --timeout 1800 --workdir $W --root runs_paper &
wait
for c in 0.55 0.9 0.96; do
  python cgcp_sarsop.py --task rocksample5_tight --c $c --timeout 1800 --ub-timeout 7200 --workdir $W --root runs_paper &
done
wait
```

Every policy found is valid for every threshold, so the thresholds share their policies through a cache in
`runs_paper/p2_cgcp/cache`. Each threshold writes `runs_paper/p2_cgcp/<task>_cgcp_c<c>_t<timeout>/result.json`. A rerun
skips finished thresholds and takes solved columns from the cache. SARSOP saves its policy and bounds when it is
interrupted once, so a call stopped early still counts.

### Simpler upper bounds

`cheap_bounds.py` computes two bounds that are cheaper than SARSOP's. Both are looser.

| Bound | RS(5,3), $`c=`$ 0.55 / 0.8 / 0.86 | RS(5,5), $`c=`$ 0.55 / 0.9 / 0.96 |
|---|---|---|
| CGCP, the value it reaches | 17.74 / 17.18 / 17.05 | 19.64 / 19.27 / 18.36 |
| SARSOP through the inequality above | 17.79 / 17.22 / 17.09 | 20.86 / 20.39 / 20.31 |
| Fast informed bound | 18.86 / 18.46 / 18.36 | 23.20 / 23.20 / 23.20 |
| The agent observes the rock qualities | 19.31 / 19.31 / 19.31 | 23.20 / 23.20 / 23.20 |

## Tables and figures

```bash
python suite.py --report                           # reports_paper/: T-maze, slack and polar-maze tables and figures, numbers.md
python plot_p2.py --no-npg --cloud-ours-only       # reports_p2/: the RockSample table, both figures below and numbers_p2.md
```
