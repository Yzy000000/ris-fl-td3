import csv
import numpy as np
import matplotlib.pyplot as plt
import os

csv_path = os.path.join(os.path.dirname(__file__), 'plots', 'training_stats.csv')
plots_dir = os.path.join(os.path.dirname(__file__), 'plots')

if not os.path.exists(csv_path):
    print('CSV not found:', csv_path)
    raise SystemExit(1)

episodes = []
rewards = []
delays = []
pb = []
pc = []
ps = []

with open(csv_path, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        episodes.append(int(row['episode']))
        rewards.append(float(row['reward']))
        # handle 'inf' strings
        d = row['avg_max_delay']
        delays.append(np.inf if d == 'inf' or d == 'infs' else float(d))
        pb.append(float(row['p_bandwidth']))
        pc.append(float(row['p_convergence']))
        ps.append(float(row['p_secrecy']))

rewards = np.array(rewards)
delays = np.array(delays)
pb = np.array(pb)
pc = np.array(pc)
ps = np.array(ps)

# compute stats for delays (ignore inf for percentiles)
finite_mask = np.isfinite(delays)
finite_delays = delays[finite_mask]

print('Episodes:', len(episodes))
print('Delay stats (finite values):')
if finite_delays.size > 0:
    print('  median:', np.median(finite_delays))
    print('  25th pct:', np.percentile(finite_delays, 25))
    print('  75th pct:', np.percentile(finite_delays, 75))
    print('  90th pct:', np.percentile(finite_delays, 90))
    print('  mean:', np.mean(finite_delays))
else:
    print('  no finite delays')

# proportion of episodes with delay < 1s, <0.5s, <5s
if len(delays) > 0:
    prop_lt_1 = np.sum((delays < 1.0) & finite_mask) / len(delays)
    prop_lt_0_5 = np.sum((delays < 0.5) & finite_mask) / len(delays)
    prop_lt_5 = np.sum((delays < 5.0) & finite_mask) / len(delays)
    print(f'Proportion episodes with avg_max_delay <1s: {prop_lt_1:.3f}')
    print(f'Proportion episodes with avg_max_delay <0.5s: {prop_lt_0_5:.3f}')
    print(f'Proportion episodes with avg_max_delay <5s: {prop_lt_5:.3f}')

# reward stats
print('\nReward stats:')
print('  median:', np.median(rewards))
print('  mean:', np.mean(rewards))
print('  min:', np.min(rewards))
print('  max:', np.max(rewards))

# penalty stats (per-episode totals)
print('\nPenalty (per-episode total) stats:')
print('  p_bandwidth mean:', np.mean(pb))
print('  p_convergence mean:', np.mean(pc))
print('  p_secrecy mean:', np.mean(ps))

# scatter plot reward vs delay (log y)
plt.figure(figsize=(6,4))
mask = finite_mask
plt.scatter(rewards[mask], finite_delays, c='C0')
plt.xlim(np.min(rewards)-1, np.max(rewards)+1)
plt.yscale('log')
plt.xlabel('Total Reward')
plt.ylabel('Avg Max Delay (s)')
plt.title('Reward vs Avg Max Delay (finite delays)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'reward_vs_delay.png'))
plt.close()

print('Saved reward_vs_delay.png to', plots_dir)
