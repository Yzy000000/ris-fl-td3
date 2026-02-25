"""
Training script for TD3 on the RLFLEnv.

This script runs a small test training loop (default 10 episodes) to verify integration.
It converts environment observations to a simple state vector:
  state = [|h_k_b| for k] + [|h_k_e| for k] + [avg_selected_placeholder]

Action mapping (from agent output in [-1,1]):
 - first K dims -> selection logits -> threshold at 0 -> binary x
 - next K dims -> bandwidth raw -> softmax to sum<=1
 - last M dims -> phases: map [-1,1] -> [0, 2*pi]
"""
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

try:
    from new_paper.env import RLFLEnv
    from new_paper.td3 import TD3Agent
    from new_paper.config import Config
except Exception:
    # fallback for direct execution
    import importlib.util, os
    base = os.path.dirname(__file__)
    spec_env = importlib.util.spec_from_file_location('new_paper.env', os.path.join(base, 'env.py'))
    mod_env = importlib.util.module_from_spec(spec_env)
    spec_env.loader.exec_module(mod_env)
    RLFLEnv = mod_env.RLFLEnv

    spec_td3 = importlib.util.spec_from_file_location('new_paper.td3', os.path.join(base, 'td3.py'))
    mod_td3 = importlib.util.module_from_spec(spec_td3)
    spec_td3.loader.exec_module(mod_td3)
    TD3Agent = mod_td3.TD3Agent

    spec_cfg = importlib.util.spec_from_file_location('new_paper.config', os.path.join(base, 'config.py'))
    mod_cfg = importlib.util.module_from_spec(spec_cfg)
    spec_cfg.loader.exec_module(mod_cfg)
    Config = mod_cfg.Config


def obs_to_state(obs, K):
    # obs['channels'] contains 'h_k_b' (K,), 'h_k_e' (K,)
    chans = obs['channels']
    h_k_b = np.asarray(chans['h_k_b']).flatten()
    h_k_e = np.asarray(chans['h_k_e']).flatten()
    # basic features: magnitudes
    feats = np.concatenate([np.abs(h_k_b), np.abs(h_k_e)])
    # append a placeholder scalar (e.g., sum of magnitudes)
    feats = np.concatenate([feats, [np.mean(np.abs(h_k_b))]])
    return feats


def map_action_to_env(action: np.ndarray, K: int, M: int, min_frac: float = None):
    # action length should be K+K+M
    a_sel = action[:K]
    a_bw = action[K:K+K]
    a_theta = action[K+K:]

    # selection: threshold at 0
    x = (a_sel > 0).astype(int)

    # bandwidth: softmax of a_bw to get fractions summing to 1, then scale down by <=1 (safe)
    exp_bw = np.exp(a_bw - np.max(a_bw))
    b_k = exp_bw / (np.sum(exp_bw) + 1e-12)

    # enforce minimum bandwidth fraction for selected users to avoid near-zero rates
    if min_frac is None:
        min_frac = 1e-2
    # if any selected have fraction < min_frac, bump them and renormalize
    if np.any(x == 1):
        small_idxs = np.where((x == 1) & (b_k < min_frac))[0]
        if small_idxs.size > 0:
            b_k[small_idxs] = min_frac
            # renormalize to sum to 1
            b_k = b_k / (np.sum(b_k) + 1e-12)

    # theta: map [-1,1] -> [0, 2*pi]
    theta = (a_theta + 1.0) / 2.0 * (2.0 * np.pi)

    return {'x': x, 'b_k': b_k, 'theta': theta}


def train(episodes: int = 10, max_steps: int = 50, seed: int = 0):
    cfg = Config()
    env = RLFLEnv(cfg, seed=seed)
    K = int(cfg.system.K)
    M = int(cfg.system.M)

    state_dim = 2 * K + 1
    action_dim = K + K + M

    # auto-select device: use CUDA if available
    try:
        import torch as _torch
        device_str = 'cuda' if _torch.cuda.is_available() else 'cpu'
    except Exception:
        device_str = 'cpu'
    print(f"Using device: {device_str}")
    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, device=device_str)

    total_rewards = []
    delay_stats = []
    all_step_delays = []
    all_step_rewards = []
    all_step_pconvs = []
    top_ratios = []
    # 动态调整 min_frac 的参数
    min_frac = 0.01  # 初始值
    min_frac_max = 0.1  # 最大不超过10%
    adapt_interval = 100  # 每100集检查一次
    top_threshold = 0.05  # 触顶比例超过5%就提高
    # track penalty sums for monitoring (per-episode)
    penalty_bandwidth_list = []
    penalty_convergence_list = []
    penalty_secrecy_list = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        state = obs_to_state(obs, K)
        ep_reward = 0.0
        ep_delays = []
        ep_hit_count = 0
        steps_run = 0
        ep_pb = 0.0
        ep_pc = 0.0
        ep_ps = 0.0
        for t in range(max_steps):
            action = agent.select_action(state, add_noise=True)
            env_action = map_action_to_env(action, K, M, min_frac=min_frac)
            next_obs, reward, done, info = env.step(env_action)
            next_state = obs_to_state(next_obs, K)
            steps_run += 1
            # debug print guarded by flag to avoid huge logs during batch runs
            PRINT_STEP = False
            if PRINT_STEP and isinstance(info, dict):
                md_dbg = info.get('max_delay', None)
                try:
                    if md_dbg is None:
                        print(f"Step {t}: info has no 'max_delay' key")
                    else:
                        print(f"Step {t}: max_delay from info = {float(md_dbg):.6f} s, reward = {float(reward):.6f}")
                except Exception:
                    print(f"Step {t}: could not print max_delay (value={md_dbg})")

            # store experience
            agent.replay.add(state.astype(np.float32), action.astype(np.float32), np.array([reward], dtype=np.float32), next_state.astype(np.float32), False)

            # update
            agent.update()

            # record delay if present and finite
            if isinstance(info, dict):
                md = info.get('max_delay', None)
                if md is not None and np.isfinite(md):
                    md_val = float(md)
                    ep_delays.append(md_val)
                    all_step_delays.append(md_val)
                    # count if hitting the cap (~>=999s)
                    if md_val >= 999.0:
                        ep_hit_count += 1
                # record per-step reward and p_convergence
                all_step_rewards.append(float(reward))
                all_step_pconvs.append(float(info.get('p_convergence', 0.0)))
                # accumulate episode penalties
                ep_pb += float(info.get('p_bandwidth', 0.0))
                ep_pc += float(info.get('p_convergence', 0.0))
                ep_ps += float(info.get('p_secrecy', 0.0))

            state = next_state
            ep_reward += reward
            if done:
                break

        total_rewards.append(ep_reward)
        if ep_delays:
            delay_stats.append(float(np.mean(ep_delays)))
        else:
            delay_stats.append(float('inf'))
        # record per-episode penalty sums
        penalty_bandwidth_list.append(ep_pb)
        penalty_convergence_list.append(ep_pc)
        penalty_secrecy_list.append(ep_ps)
        # per-episode top ratio (fraction of steps hitting cap)
        ep_top_ratio = float(ep_hit_count) / float(max(1, steps_run))
        top_ratios.append(ep_top_ratio)
        print(f'Episode {ep+1}/{episodes} reward={ep_reward:.3f}')
        print(f' Episode {ep+1}/{episodes} avg_max_delay={delay_stats[-1]:.6f}s')

        # print penalty averages every 10 episodes
        if (ep + 1) % 10 == 0:
            avg_pb = float(np.mean(penalty_bandwidth_list))
            avg_pc = float(np.mean(penalty_convergence_list))
            avg_ps = float(np.mean(penalty_secrecy_list))
            avg_pb_step = avg_pb / float(max_steps)
            avg_pc_step = avg_pc / float(max_steps)
            avg_ps_step = avg_ps / float(max_steps)
            print(f' Penalty averages per-episode up to ep {ep+1}: p_bandwidth_total={avg_pb:.6f}, p_convergence_total={avg_pc:.6f}, p_secrecy_total={avg_ps:.6f}')
            print(f' Penalty averages per-step up to ep {ep+1}: p_bandwidth={avg_pb_step:.6f}, p_convergence={avg_pc_step:.6f}, p_secrecy={avg_ps_step:.6f}')

        # adaptive min_frac adjustment every adapt_interval episodes
        if (ep + 1) > 0 and (ep + 1) % adapt_interval == 0:
            recent_top_ratio = float(np.mean(top_ratios[-adapt_interval:])) if len(top_ratios) >= adapt_interval else float(np.mean(top_ratios))
            if recent_top_ratio > top_threshold:
                new_min = min(min_frac * 1.5, min_frac_max)
                if new_min > min_frac:
                    min_frac = new_min
                    env.min_frac = min_frac
                    print(f"Episode {ep+1}: Increasing min_frac to {min_frac:.3f} (top_ratio={recent_top_ratio:.2%})")

    # quick save
    out_dir = os.path.join('new_paper', 'models')
    os.makedirs(out_dir, exist_ok=True)
    agent.save(os.path.join(out_dir, 'td3_risfl'))
    print('Training finished. Models saved to', out_dir)

    # --- visualization and logging ---
    plots_dir = os.path.join('new_paper', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # save CSV of episode stats
    csv_path = os.path.join(plots_dir, 'training_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'avg_max_delay', 'p_bandwidth', 'p_convergence', 'p_secrecy'])
        for i in range(len(total_rewards)):
            writer.writerow([i+1, total_rewards[i], delay_stats[i], penalty_bandwidth_list[i], penalty_convergence_list[i], penalty_secrecy_list[i]])

    # plot rewards
    plt.figure()
    plt.plot(range(1, len(total_rewards)+1), total_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'reward_per_episode.png'))
    plt.close()

    # plot avg delay
    plt.figure()
    plt.plot(range(1, len(delay_stats)+1), delay_stats, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Avg Max Delay (s)')
    plt.yscale('log')
    plt.title('Average Max Delay per Episode (log scale)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'avg_delay_per_episode.png'))
    plt.close()

    # plot penalties (per-episode totals)
    plt.figure()
    plt.plot(range(1, len(penalty_bandwidth_list)+1), penalty_bandwidth_list, label='bandwidth')
    plt.plot(range(1, len(penalty_convergence_list)+1), penalty_convergence_list, label='convergence')
    plt.plot(range(1, len(penalty_secrecy_list)+1), penalty_secrecy_list, label='secrecy')
    plt.xlabel('Episode')
    plt.ylabel('Penalty (per-episode sum)')
    plt.yscale('log')
    plt.title('Per-episode Penalties (log scale)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'penalties_per_episode.png'))
    plt.close()

    print('Plots and CSV saved to', plots_dir)

    # --- compute and print requested summary statistics ---
    import numpy as _np
    if len(all_step_delays) > 0:
        delays_arr = _np.array(all_step_delays)
        delay_median = float(_np.median(delays_arr))
        delay_mean = float(_np.mean(delays_arr))
        delay_p90 = float(_np.percentile(delays_arr, 90))
        # count steps hitting the cap ~1000s
        hit_count = int(_np.sum(delays_arr >= 999.0))
        hit_prop = float(hit_count) / float(len(delays_arr))
    else:
        delay_median = delay_mean = delay_p90 = float('nan')
        hit_prop = 0.0

    # reward median (per-episode)
    reward_median_episode = float(_np.median(_np.array(total_rewards))) if total_rewards else float('nan')

    # p_convergence mean (per-step)
    pconv_mean = float(_np.mean(_np.array(all_step_pconvs))) if all_step_pconvs else float('nan')

    print('\n=== Quick Summary ===')
    print(f'Delay median={delay_median:.6f}s, mean={delay_mean:.6f}s, 90th={delay_p90:.6f}s')
    print(f'Reward median (per-episode)={reward_median_episode:.6f}')
    print(f'p_convergence mean (per-step)={pconv_mean:.6f}')
    print(f'Proportion of steps hitting cap (>=999s)={hit_prop:.6%} ({hit_count}/{len(all_step_delays)})')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)
