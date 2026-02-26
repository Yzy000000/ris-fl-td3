#!/usr/bin/env python
"""
Generate all paper-style figures from training_stats.csv
Run from workspace root: python generate_all_figures.py
"""
import os
import sys

# Change to plots directory
os.chdir(os.path.join(os.path.dirname(__file__), 'new_paper', 'plots'))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

print("=" * 60)
print("GENERATING PAPER-STYLE FIGURES")
print("=" * 60)

# Read CSV
print("\n1. Reading training_stats.csv...")
df = pd.read_csv('training_stats.csv')
print(f"   Loaded {len(df)} episodes")

# === FIGURE 2: Reward and Delay Curves ===
print("\n2. Generating Figure 2 (Reward & Delay Curves)...")
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Reward curve
ax1 = axes[0]
episodes = df['episode'].values
rewards = df['reward'].values
ax1.plot(episodes, rewards, alpha=0.3, color='blue', linewidth=1, label='Raw Reward')
if len(rewards) > 50:
    smoothed = savgol_filter(rewards, window_length=51, polyorder=3)
    ax1.plot(episodes, smoothed, color='red', linewidth=2, label='Smoothed Reward')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Training Reward Curve')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Delay curve
ax2 = axes[1]
delays = df['avg_max_delay'].values
ax2.plot(episodes, delays, alpha=0.3, color='green', linewidth=1, label='Raw Delay')
if len(delays) > 50:
    delayed_smoothed = savgol_filter(delays, window_length=51, polyorder=3)
    ax2.plot(episodes, delayed_smoothed, color='darkgreen', linewidth=2, label='Smoothed Delay')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Average Max Delay (s)')
ax2.set_title('Delay Convergence')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_yscale('log')
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('figure2_reward_delay_curves.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_reward_delay_curves.pdf', bbox_inches='tight')
plt.close()
print("   ✓ Saved: figure2_reward_delay_curves.png/pdf")

# === FIGURE 3: Delay Distribution ===
print("\n3. Generating Figure 3 (Delay Distribution)...")
early = df[df['episode'] <= 100]['avg_max_delay']
middle = df[(df['episode'] > 100) & (df['episode'] <= 300)]['avg_max_delay']
late = df[df['episode'] > 300]['avg_max_delay']

fig, ax = plt.subplots(figsize=(8, 6))
data_to_plot = [early, middle, late]
positions = [1, 2, 3]
labels = ['Early (1-100)', 'Middle (101-300)', 'Late (301-500)']

bp = ax.boxplot(data_to_plot, positions=positions, patch_artist=True,
                widths=0.6, showfliers=False,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

for i, data in enumerate(data_to_plot):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    y = outliers.values
    x = np.random.normal(positions[i], 0.08, size=len(y))
    ax.scatter(x, y, alpha=0.5, s=20, color='gray')

ax.set_xticklabels(labels)
ax.set_ylabel('Average Max Delay (s)')
ax.set_title('Delay Distribution Across Training Stages')
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')
ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5)

stats_text = f"""Late Stage Statistics:
Median: {late.median():.2f}s
Mean: {late.mean():.2f}s
90th: {late.quantile(0.9):.2f}s
% < 10s: {(late < 10).mean()*100:.1f}%"""
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figure3_delay_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_delay_distribution.pdf', bbox_inches='tight')
plt.close()
print("   ✓ Saved: figure3_delay_distribution.png/pdf")

# === FIGURE 4: Penalties ===
print("\n4. Generating Figure 4 (Penalty Analysis)...")
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

penalties = [
    ('p_convergence', 'Convergence Penalty', 'blue'),
    ('p_secrecy', 'Secrecy Penalty', 'red'),
    ('p_bandwidth', 'Bandwidth Penalty', 'green')
]

for ax, (col, title, color) in zip(axes, penalties):
    episodes = df['episode'].values
    values = df[col].values
    ax.plot(episodes, values, alpha=0.3, color=color, linewidth=1, label=f'Raw {title}')
    
    window = 20
    if len(values) > window:
        ma = np.convolve(values, np.ones(window)/window, mode='same')
        ax.plot(episodes, ma, color=color, linewidth=2, label=f'Moving Average ({window} eps)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(title)
    ax.set_title(f'{title} Over Training')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    mean_val = values[-100:].mean()
    ax.text(0.02, 0.95, f'Last 100 eps mean: {mean_val:.3f}', 
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figure_penalties.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_penalties.pdf', bbox_inches='tight')
plt.close()
print("   ✓ Saved: figure_penalties.png/pdf")

print("\n" + "=" * 60)
print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
print("\nGenerated files:")
for f in ['figure2_reward_delay_curves.png', 'figure3_delay_distribution.png', 'figure_penalties.png']:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"  ✓ {f} ({size:,} bytes)")
