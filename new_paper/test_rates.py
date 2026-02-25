import numpy as np
import os
try:
    from new_paper.config import Config
    from new_paper.channel import ChannelGenerator
    from new_paper.secrecy import compute_secrecy_rates, compute_effective_channel
except Exception:
    import importlib.util
    base = os.path.dirname(__file__)
    spec_cfg = importlib.util.spec_from_file_location('new_paper.config', os.path.join(base, 'config.py'))
    cfg_mod = importlib.util.module_from_spec(spec_cfg)
    spec_cfg.loader.exec_module(cfg_mod)
    Config = cfg_mod.Config

    spec_ch = importlib.util.spec_from_file_location('new_paper.channel', os.path.join(base, 'channel.py'))
    ch_mod = importlib.util.module_from_spec(spec_ch)
    spec_ch.loader.exec_module(ch_mod)
    ChannelGenerator = ch_mod.ChannelGenerator

    spec_s = importlib.util.spec_from_file_location('new_paper.secrecy', os.path.join(base, 'secrecy.py'))
    s_mod = importlib.util.module_from_spec(spec_s)
    spec_s.loader.exec_module(s_mod)
    compute_secrecy_rates = s_mod.compute_secrecy_rates
    compute_effective_channel = s_mod.compute_effective_channel

cfg = Config()
K = int(cfg.system.K)
M = int(cfg.system.M)

gen = ChannelGenerator(cfg, seed=0)
gen.reset(seed=1)
chans = gen.generate_channels(return_torch=False)

# construct action: all selected, uniform bandwidth, random phases
x = np.ones(K, dtype=int)
b_k = np.ones(K) / float(K)
theta = np.random.uniform(-1, 1, M)
# env would convert real theta to exp(1j*theta); mimic that
theta_complex = np.exp(1j * theta)

participant_indices = list(range(K))
jammer_indices = []  # none

rates_info, sats = compute_secrecy_rates(theta_complex, participant_indices, jammer_indices, chans, b_k, cfg, backend='numpy')

print('Config B (Hz):', cfg.channel.B)
print('S_k bits:', cfg.fl.S_k_bits)
print('sigma2_b:', cfg.channel.sigma2_b)
print('P_k:', cfg.system.P_k)
print('\nPer-participant rates and SINR:')
for k in participant_indices:
    info = rates_info[k]
    Rkb = info['R_kb']
    Rks = info['R_ks']
    # compute SINR manually
    # compute g_kb and g_ke
    gkb = compute_effective_channel(theta_complex, chans['h_r_b'], chans['h_k_r'][k], chans['h_k_b'][k], backend='numpy')
    gke = compute_effective_channel(theta_complex, chans['h_r_e'], chans['h_k_r'][k], chans['h_k_e'][k], backend='numpy')
    # interference: sum over j not in participants -> zero here
    Ib = 0.0
    Ie = 0.0
    P = float(cfg.system.P_k)
    sigma2_b = float(cfg.channel.sigma2_b)
    sigma2_e = float(cfg.channel.sigma2_e)
    gamma_b = (P * gkb) / (Ib + sigma2_b)
    gamma_e = (P * gke) / (Ie + sigma2_e)
    # expected Rkb computed as b_k[k] * B * log2(1+gamma_b)
    expected_Rkb = float(b_k[k]) * float(cfg.channel.B) * np.log2(1.0 + gamma_b)
    print(f' k={k}: gkb={gkb:.3e}, gamma_b={gamma_b:.3e}, Rkb={Rkb:.3f}, expected_Rkb={expected_Rkb:.3f}, Rks={Rks:.3f}')
    T_tr = cfg.fl.S_k_bits / max(Rkb, 1e-9)
    print(f'    T_tr (s) = {T_tr:.3e}')

# print some channel norms
print('\nChannel norms:')
print('h_r_b norm:', np.linalg.norm(chans['h_r_b']))
print('h_r_e norm:', np.linalg.norm(chans['h_r_e']))
print('h_k_r norms:', np.linalg.norm(chans['h_k_r'], axis=1))
print('h_k_b abs:', np.abs(chans['h_k_b']))
print('h_k_e abs:', np.abs(chans['h_k_e']))
