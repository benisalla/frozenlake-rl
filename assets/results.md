### frozenlake-4x4

| method | family | episodes | env steps | episodes to success >= 1 | exact success (greedy) | rollout success | rollout mean return | train time |
|---|---|---|---|---|---|---|---|---|
| `value-iteration` | DP | model | - | 0 (6 sweeps) | 1.000 | 1.000 | 1.00 | 0.0 s |
| `policy-iteration` | DP | model | - | 0 (6 sweeps) | 1.000 | 1.000 | 1.00 | 0.0 s |
| `mc-first-visit` | MC | 1,000 | 8,658 | 40 | 1.000 | 1.000 | 1.00 | 0.2 s |
| `mc-every-visit` | MC | 1,000 | 7,827 | 40 | 1.000 | 1.000 | 1.00 | 0.2 s |
| `sarsa` | TD | 1,000 | 8,182 | 160 | 1.000 | 1.000 | 1.00 | 0.2 s |
| `expected-sarsa` | TD | 1,000 | 6,848 | 80 | 1.000 | 1.000 | 1.00 | 0.3 s |
| `q-learning` | TD | 1,000 | 7,055 | 80 | 1.000 | 1.000 | 1.00 | 0.2 s |
| `double-q-learning` | TD | 1,000 | 6,692 | 260 | 1.000 | 1.000 | 1.00 | 0.2 s |
| `n-step-sarsa` | TD | 1,000 | 8,368 | 140 | 1.000 | 1.000 | 1.00 | 0.2 s |
| `sarsa-lambda` | TD | 1,000 | 7,745 | 40 | 1.000 | 1.000 | 1.00 | 0.2 s |
| `q-lambda` | TD | 1,000 | 6,922 | 40 | 1.000 | 1.000 | 1.00 | 0.2 s |
| `dyna-q` | planning | 1,000 | 6,443 | 40 | 1.000 | 1.000 | 1.00 | 1.4 s |
| `dqn` | neural | 1,000 | 8,906 | 260 | 1.000 | 1.000 | 1.00 | 4.3 s |
| `reinforce` | neural | 1,000 | 7,009 | 380 | 1.000 | 1.000 | 1.00 | 0.5 s |

best possible success within the time limit (finite-horizon DP): 1.000

suite wall time: 0.3 min

### frozenlake-4x4-slippery

| method | family | episodes | env steps | episodes to success >= 0.7 | exact success (greedy) | rollout success | rollout mean return | train time |
|---|---|---|---|---|---|---|---|---|
| `value-iteration` | DP | model | - | 0 (12 sweeps) | 0.740 | 0.729 | 0.73 | 0.2 s |
| `policy-iteration` | DP | model | - | 0 (4 sweeps) | 0.740 | 0.729 | 0.73 | 0.0 s |
| `mc-first-visit` | MC | 20,000 | 451,466 | 6,400 | 0.740 | 0.729 | 0.73 | 7.8 s |
| `mc-every-visit` | MC | 20,000 | 254,744 | 19,200 | 0.728 | 0.724 | 0.72 | 4.4 s |
| `sarsa` | TD | 10,000 | 240,232 | 4,000 | 0.730 | 0.728 | 0.73 | 4.4 s |
| `expected-sarsa` | TD | 10,000 | 239,497 | 2,800 | 0.730 | 0.728 | 0.73 | 7.3 s |
| `q-learning` | TD | 10,000 | 258,598 | 1,400 | 0.740 | 0.729 | 0.73 | 5.5 s |
| `double-q-learning` | TD | 10,000 | 238,703 | 3,600 | 0.730 | 0.726 | 0.73 | 3.9 s |
| `n-step-sarsa` | TD | 10,000 | 222,946 | 5,400 | 0.730 | 0.728 | 0.73 | 5.1 s |
| `sarsa-lambda` | TD | 10,000 | 232,770 | 4,600 | 0.740 | 0.729 | 0.73 | 5.7 s |
| `q-lambda` | TD | 10,000 | 265,673 | 2,000 | 0.728 | 0.724 | 0.72 | 6.6 s |
| `dyna-q` | planning | 2,000 | 51,351 | 240 | 0.740 | 0.729 | 0.73 | 10.6 s |
| `dqn` | neural | 1,500 | 36,754 | 840 | 0.728 | 0.724 | 0.72 | 8.7 s |
| `reinforce` | neural | 8,000 | 173,369 | 4,320 | 0.730 | 0.726 | 0.73 | 5.7 s |

best possible success within the time limit (finite-horizon DP): 0.744

suite wall time: 1.5 min

### frozenlake-8x8

| method | family | episodes | env steps | episodes to success >= 1 | exact success (greedy) | rollout success | rollout mean return | train time |
|---|---|---|---|---|---|---|---|---|
| `value-iteration` | DP | model | - | 0 (14 sweeps) | 1.000 | 1.000 | 1.00 | 0.0 s |
| `policy-iteration` | DP | model | - | 0 (14 sweeps) | 1.000 | 1.000 | 1.00 | 0.0 s |
| `mc-first-visit` | MC | 3,000 | 78,514 | 1,740 | 1.000 | 1.000 | 1.00 | 1.8 s |
| `mc-every-visit` | MC | 3,000 | 105,934 | 1,740 | 1.000 | 1.000 | 1.00 | 2.3 s |
| `sarsa` | TD | 3,000 | 80,374 | 1,980 | 1.000 | 1.000 | 1.00 | 2.0 s |
| `expected-sarsa` | TD | 3,000 | 78,163 | 1,980 | 1.000 | 1.000 | 1.00 | 3.3 s |
| `q-learning` | TD | 3,000 | 80,339 | 1,980 | 1.000 | 1.000 | 1.00 | 2.1 s |
| `double-q-learning` | TD | 3,000 | 63,818 | 1,200 | 1.000 | 1.000 | 1.00 | 1.0 s |
| `n-step-sarsa` | TD | 3,000 | 76,636 | 1,680 | 1.000 | 1.000 | 1.00 | 1.5 s |
| `sarsa-lambda` | TD | 3,000 | 75,464 | 1,680 | 1.000 | 1.000 | 1.00 | 1.7 s |
| `q-lambda` | TD | 3,000 | 75,118 | 1,680 | 1.000 | 1.000 | 1.00 | 1.7 s |
| `dyna-q` | planning | 3,000 | 52,060 | 300 | 1.000 | 1.000 | 1.00 | 8.6 s |
| `dqn` | neural | 2,000 | 55,455 | 440 | 1.000 | 1.000 | 1.00 | 19.9 s |
| `reinforce` | neural | 4,000 | 101,479 | 80 | 1.000 | 1.000 | 1.00 | 4.2 s |

best possible success within the time limit (finite-horizon DP): 1.000

suite wall time: 1.0 min

### frozenlake-8x8-slippery

| method | family | episodes | env steps | episodes to success >= 0.8 | exact success (greedy) | rollout success | rollout mean return | train time |
|---|---|---|---|---|---|---|---|---|
| `value-iteration` | DP | model | - | 0 (39 sweeps) | 0.863 | 0.874 | 0.87 | 0.8 s |
| `policy-iteration` | DP | model | - | 0 (8 sweeps) | 0.863 | 0.874 | 0.87 | 0.0 s |
| `mc-first-visit` | MC | 40,000 | 1,972,555 | not reached | 0.619 | 0.620 | 0.62 | 49.2 s |
| `mc-every-visit` | MC | 40,000 | 1,448,913 | not reached | 0.031 | 0.043 | 0.04 | 40.9 s |
| `sarsa` | TD | 20,000 | 1,331,978 | 8,800 | 0.857 | 0.863 | 0.86 | 17.3 s |
| `expected-sarsa` | TD | 20,000 | 1,320,833 | 8,400 | 0.857 | 0.865 | 0.86 | 39.8 s |
| `q-learning` | TD | 20,000 | 1,284,219 | 5,600 | 0.848 | 0.870 | 0.87 | 21.5 s |
| `double-q-learning` | TD | 20,000 | 1,231,597 | 6,400 | 0.827 | 0.847 | 0.85 | 22.7 s |
| `n-step-sarsa` | TD | 20,000 | 1,327,470 | 14,000 | 0.879 | 0.904 | 0.90 | 47.7 s |
| `sarsa-lambda` | TD | 20,000 | 1,231,090 | 16,000 | 0.845 | 0.855 | 0.85 | 26.6 s |
| `q-lambda` | TD | 20,000 | 1,312,515 | 4,800 | 0.844 | 0.842 | 0.84 | 55.5 s |
| `dyna-q` | planning | 2,500 | 131,500 | 1,600 | 0.875 | 0.893 | 0.89 | 39.7 s |
| `dqn` | neural | 2,000 | 133,205 | 1,000 | 0.841 | 0.850 | 0.85 | 69.3 s |
| `reinforce` | neural | 15,000 | 911,061 | 7,500 | 0.850 | 0.869 | 0.87 | 41.2 s |

best possible success within the time limit (finite-horizon DP): 0.913

suite wall time: 8.3 min

### taxi

| method | family | episodes | env steps | episodes to return >= 7 | exact return (greedy) | rollout success | rollout mean return | train time |
|---|---|---|---|---|---|---|---|---|
| `value-iteration` | DP | model | - | 0 (18 sweeps) | 7.93 | 1.000 | 7.91 | 0.7 s |
| `policy-iteration` | DP | model | - | 0 (16 sweeps) | 7.93 | 1.000 | 7.91 | 1.1 s |
| `sarsa` | TD | 3,000 | 149,508 | 2,340 | 7.62 | 1.000 | 7.61 | 5.1 s |
| `expected-sarsa` | TD | 3,000 | 130,943 | 1,860 | 7.92 | 1.000 | 7.90 | 7.0 s |
| `q-learning` | TD | 3,000 | 110,125 | 1,260 | 7.89 | 1.000 | 7.86 | 4.4 s |
| `double-q-learning` | TD | 3,000 | 130,287 | 2,400 | 7.93 | 1.000 | 7.91 | 5.0 s |
| `n-step-sarsa` | TD | 5,000 | 218,902 | 3,000 | 7.78 | 1.000 | 7.76 | 7.7 s |
| `sarsa-lambda` | TD | 3,000 | 132,939 | 2,220 | 7.64 | 1.000 | 7.61 | 6.1 s |
| `q-lambda` | TD | 3,000 | 109,056 | 720 | 7.86 | 1.000 | 7.87 | 3.8 s |
| `dyna-q` | planning | 1,000 | 35,159 | 120 | 7.93 | 1.000 | 7.91 | 7.1 s |

best possible return within the time limit (finite-horizon DP): 7.930

suite wall time: 1.0 min

### cliffwalking

| method | family | episodes | env steps | episodes to return >= -13 | exact return (greedy) | rollout success | rollout mean return | train time |
|---|---|---|---|---|---|---|---|---|
| `value-iteration` | DP | model | - | 0 (13 sweeps) | -13.00 | 1.000 | -13.00 | 0.0 s |
| `policy-iteration` | DP | model | - | 0 (12 sweeps) | -13.00 | 1.000 | -13.00 | 1.0 s |
| `sarsa` | TD | 500 | 12,459 | not reached | -17.00 | 1.000 | -17.00 | 0.2 s |
| `expected-sarsa` | TD | 500 | 10,547 | not reached | -15.00 | 1.000 | -15.00 | 0.4 s |
| `q-learning` | TD | 500 | 10,348 | 60 | -13.00 | 1.000 | -13.00 | 0.2 s |
| `double-q-learning` | TD | 500 | 15,272 | not reached | -15.00 | 1.000 | -15.00 | 0.3 s |
| `n-step-sarsa` | TD | 500 | 18,513 | not reached | -19.00 | 1.000 | -19.00 | 0.4 s |
| `sarsa-lambda` | TD | 500 | 15,349 | not reached | -17.00 | 1.000 | -17.00 | 0.4 s |
| `q-lambda` | TD | 500 | 11,208 | not reached | -17.00 | 1.000 | -17.00 | 0.4 s |
| `dyna-q` | planning | 500 | 9,001 | 20 | -13.00 | 1.000 | -13.00 | 2.1 s |

best possible return within the time limit (finite-horizon DP): -13.000

suite wall time: 0.2 min
