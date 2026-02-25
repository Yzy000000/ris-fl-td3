"""
Secrecy rate computation for RIS-assisted FL (based on Wu et al. 2025 Sec V-A, eq. (8)-(10)).

提供函数：
- `compute_effective_channel(theta, h_r, h_k_r, h_k_direct, backend='torch')` 返回标量通道功率 |h_r^H Θ h_k_r + h_k_direct|^2
- `compute_secrecy_rates(theta, participant_indices, jammer_indices, channels, b_k, config, backend='torch')`

函数优先使用 `torch`（若安装），否则回退到 `numpy`。

说明：输入 `channels` 支持 `h_k_r` 为形状 (K,M) 的 numpy array 或 list-of-(M,) arrays。
"""
from typing import Dict, List, Tuple, Union
import numpy as np


def _has_torch() -> bool:
    try:
        import torch
        return True
    except Exception:
        return False


def _to_complex_tensor(x, use_torch: bool):
    if use_torch:
        import torch
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.complex64))
        else:
            return torch.tensor(x, dtype=torch.complex64)
    else:
        return np.array(x, dtype=np.complex128)


def compute_effective_channel(theta: Union[np.ndarray, List[complex]],
                              h_r: Union[np.ndarray, List[complex]],
                              h_k_r: Union[np.ndarray, List[complex]],
                              h_k_direct: complex,
                              backend: str = 'torch'):
    """计算 |h_r^H Θ h_k_r + h_k_direct|^2。

    theta: (M,) 复数向量（若为实数角则传入 exp(1j*angle)）
    h_r: (M,) 复数向量
    h_k_r: (M,) 复数向量
    h_k_direct: 复标量
    backend: 'torch' 或 'numpy'
    返回：标量（torch.tensor 或 numpy.scalar），表示功率增益
    """
    use_torch = (backend == 'torch') and _has_torch()

    if use_torch:
        import torch
        theta_t = _to_complex_tensor(theta, True)
        h_r_t = _to_complex_tensor(h_r, True)
        h_k_r_t = _to_complex_tensor(h_k_r, True)
        # elementwise multiply theta * h_k_r
        prod = theta_t * h_k_r_t
        # inner product h_r^H prod
        inner = torch.sum(torch.conj(h_r_t) * prod)
        total = inner + torch.tensor(h_k_direct, dtype=torch.complex64)
        power = torch.real(torch.conj(total) * total)
        return power
    else:
        theta_n = _to_complex_tensor(theta, False)
        h_r_n = _to_complex_tensor(h_r, False)
        h_k_r_n = _to_complex_tensor(h_k_r, False)
        prod = theta_n * h_k_r_n
        inner = np.vdot(h_r_n, prod)  # conj(h_r) @ prod
        total = inner + complex(h_k_direct)
        power = np.real(np.conj(total) * total)
        return power


def compute_secrecy_rates(theta: Union[np.ndarray, List[complex]],
                          participant_indices: List[int],
                          jammer_indices: List[int],
                          channels: Dict[str, Union[np.ndarray, List[np.ndarray]]],
                          b_k: Union[List[float], np.ndarray],
                          config,
                          backend: str = 'torch') -> Tuple[Dict[int, Dict[str, float]], List[bool]]:
    """计算每个参与者的到 BS 速率 R_kb 和保密速率 R_ks，并判断是否满足 R_min。

    返回： (rates_info, satisfies_list)
      - rates_info: {k: {'R_kb': float, 'R_ks': float}}
      - satisfies_list: list of booleans，与 participant_indices 顺序对应
    """
    use_torch = (backend == 'torch') and _has_torch()
    tiny = 1e-12

    # 读取配置参数（支持多个字段名回退）
    P_device = getattr(config.system, 'P_k', None) or getattr(config, 'P_device', None) or getattr(config, 'P_k', None)
    if P_device is None:
        # fallback to fl? (some configs use system.P_k)
        P_device = 0.1
    B_total = getattr(config.channel, 'B', None) or getattr(config, 'B', None) or getattr(config, 'fl', {}).get('B', None)
    if B_total is None:
        # fallback to 1e6
        B_total = getattr(config, 'B', 1e6)
    sigma2_b = getattr(config.channel, 'sigma2_b', None) or getattr(config, 'sigma2_b', None)
    sigma2_e = getattr(config.channel, 'sigma2_e', None) or getattr(config, 'sigma2_e', None)
    R_min = getattr(config.fl, 'R_min', None) or getattr(config, 'R_min', None) or getattr(config, 'R_min', 0)

    # normalize b_k to array
    b_k_arr = np.array(b_k, dtype=float)

    # helpers to access channel elements: support h_k_r as (K,M) array or list
    h_k_r_all = channels['h_k_r']
    h_r_b = channels['h_r_b']
    h_r_e = channels['h_r_e']
    h_k_b_all = channels['h_k_b']
    h_k_e_all = channels['h_k_e']

    rates_info = {}
    satisfies = []

    # Precompute interference sums
    # For BS and Eve, sum over jammer_indices of P_device * |...|^2
    Ib = 0.0
    Ie = 0.0
    # use backend computations per-jammer
    for j in jammer_indices:
        # get h_j_r (M,)
        if isinstance(h_k_r_all, np.ndarray):
            h_j_r = h_k_r_all[j]
        else:
            h_j_r = h_k_r_all[j]
        h_j_b = h_k_b_all[j]
        h_j_e = h_k_e_all[j]
        val_b = compute_effective_channel(theta, h_r_b, h_j_r, h_j_b, backend=backend)
        val_e = compute_effective_channel(theta, h_r_e, h_j_r, h_j_e, backend=backend)
        Ib = Ib + float(P_device) * float(val_b)
        Ie = Ie + float(P_device) * float(val_e)

    # ensure scalars
    Ib = float(Ib)
    Ie = float(Ie)

    for idx in participant_indices:
        if isinstance(h_k_r_all, np.ndarray):
            h_k_r = h_k_r_all[idx]
        else:
            h_k_r = h_k_r_all[idx]
        h_k_b = h_k_b_all[idx]
        h_k_e = h_k_e_all[idx]

        g_k_b = compute_effective_channel(theta, h_r_b, h_k_r, h_k_b, backend=backend)
        g_k_e = compute_effective_channel(theta, h_r_e, h_k_r, h_k_e, backend=backend)

        # convert to floats or tensors depending on backend
        if use_torch:
            import torch
            P = torch.tensor(float(P_device), dtype=torch.float32)
            Ib_t = torch.tensor(Ib + tiny, dtype=torch.float32)
            Ie_t = torch.tensor(Ie + tiny, dtype=torch.float32)
            gkb_t = g_k_b.to(torch.float32) if torch.is_tensor(g_k_b) else torch.tensor(float(g_k_b), dtype=torch.float32)
            gke_t = g_k_e.to(torch.float32) if torch.is_tensor(g_k_e) else torch.tensor(float(g_k_e), dtype=torch.float32)
            gamma_b = (P * gkb_t) / (Ib_t + torch.tensor(float(sigma2_b), dtype=torch.float32))
            gamma_e = (P * gke_t) / (Ie_t + torch.tensor(float(sigma2_e), dtype=torch.float32))
            Rkb_t = float(b_k_arr[idx]) * float(B_total) * torch.log2(1.0 + gamma_b)
            Rke_t = float(b_k_arr[idx]) * float(B_total) * torch.log2(1.0 + gamma_e)
            Rks_t = torch.clamp(Rkb_t - Rke_t, min=0.0)
            Rkb = float(Rkb_t.detach().cpu().item())
            Rks = float(Rks_t.detach().cpu().item())
            # clip very small Rkb to avoid infinite T_tr upstream
            # raise the floor to reduce occurrences of near-zero rates
            min_Rkb = 1e-2
            if Rkb < min_Rkb:
                try:
                    print(f'Warning: R_kb for idx {idx} clipped from {Rkb} to {min_Rkb}')
                except Exception:
                    pass
                Rkb = min_Rkb
                Rks = max(0.0, Rkb - float(Rke_t.detach().cpu().item())) if 'Rke_t' in locals() else max(0.0, Rkb - 0.0)
            rates_info[idx] = {'R_kb': Rkb, 'R_ks': Rks}
            satisfies.append(Rks >= float(R_min))
        else:
            # numpy path
            gkb = float(g_k_b)
            gke = float(g_k_e)
            gamma_b = (float(P_device) * gkb) / (Ib + float(sigma2_b) + tiny)
            gamma_e = (float(P_device) * gke) / (Ie + float(sigma2_e) + tiny)
            Rkb = float(b_k_arr[idx]) * float(B_total) * np.log2(1.0 + gamma_b)
            Rke = float(b_k_arr[idx]) * float(B_total) * np.log2(1.0 + gamma_e)
            # clip very small Rkb to avoid infinite T_tr upstream
            # raise the floor to reduce occurrences of near-zero rates
            min_Rkb = 1e-2
            if Rkb < min_Rkb:
                try:
                    print(f'Warning: R_kb for idx {idx} clipped from {Rkb} to {min_Rkb}')
                except Exception:
                    pass
                Rkb = min_Rkb
            Rks = max(0.0, Rkb - Rke)
            rates_info[idx] = {'R_kb': Rkb, 'R_ks': Rks}
            satisfies.append(Rks >= float(R_min))

    return rates_info, satisfies


if __name__ == '__main__':
    # 简单自测：使用 ChannelGenerator 生成信道并计算保密速率
    try:
        from new_paper.config import Config
        from new_paper.channel import ChannelGenerator
    except Exception:
        import importlib.util, os
        base = os.path.dirname(__file__)
        spec = importlib.util.spec_from_file_location('new_paper.config', os.path.join(base, 'config.py'))
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        Config = cfg_mod.Config
        spec2 = importlib.util.spec_from_file_location('new_paper.channel', os.path.join(base, 'channel.py'))
        ch_mod = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(ch_mod)
        ChannelGenerator = ch_mod.ChannelGenerator

    cfg = Config()
    gen = ChannelGenerator(cfg, seed=0)
    gen.reset(seed=1)
    chans = gen.generate_channels(return_torch=False)
    K = cfg.system.K
    M = cfg.system.M

    # uniform bandwidth allocation
    b_k = np.ones(K) / float(K)

    # theta: all ones (no phase shift)
    theta = np.ones(M, dtype=np.complex128)

    participant_indices = list(range(K))
    # pick one jammer (last device) for test
    jammer_indices = [K-1]

    rates, sats = compute_secrecy_rates(theta, participant_indices, jammer_indices, chans, b_k, cfg, backend='numpy')
    print('secrecy rates (numpy):')
    for k in participant_indices:
        print(f'k={k}: R_s={rates[k]:.3f} bps, sat={sats[k]}')
