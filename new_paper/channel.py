"""
更精确的信道生成器，实现 Wu et al. (2025) Sec. V-A 中描述的模型。

功能：
- `ChannelGenerator` 类：管理设备位置（`reset()`）并生成信道（`generate_channels()`）。
- 使用 ULA 阵列响应作为 RIS 的 LoS 分量：a_m(phi)=exp(j*pi*m*sin(phi)), m=0..M-1（d/λ=0.5）。
- 路径损耗按 sqrt(eps * d^{-beta}) 缩放最终信道。Rician 混合由 κ 控制。

返回数据结构：
- 'h_k_r': list of K arrays each shape (M,)  (device -> RIS)
- 'h_r_b': array shape (M,)  (RIS -> BS)
- 'h_r_e': array shape (M,)  (RIS -> Eve)
- 'h_k_b': list of K complex scalars (device -> BS)
- 'h_k_e': list of K complex scalars (device -> Eve)

简化与说明：
- 直射 LoS 分量取 1（论文中可用相位项，但 λ 未指定）。
- 对于 LoS 到达角，使用从 RIS 指向目标点的角度（与 RIS 法线 x 轴的夹角），计算导向矢量。
"""

from typing import Dict, List, Optional
import numpy as np


def _safe_distance(a: np.ndarray, b: np.ndarray, min_dist: float = 1.0) -> float:
    d = np.linalg.norm(a - b)
    return max(d, min_dist)


def ula_response(M: int, phi: float) -> np.ndarray:
    """Uniform linear array response for RIS (沿 y 轴布置，法线指向 x 轴正方向).

    a_m(phi) = exp(j * pi * m * sin(phi)), m=0..M-1
    phi: angle in radians, measured from RIS 法线 (x 轴正方向)."""
    m = np.arange(M)
    return np.exp(1j * np.pi * m * np.sin(phi))


def _rician_vector_from_components(los_vec: np.ndarray, M: int, kappa: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    """根据 LoS 向量、κ 和缩放系数生成 Rician 向量（长度 M）。"""
    nlos = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2)
    if kappa <= 0:
        combined = nlos
    else:
        w_los = np.sqrt(kappa / (kappa + 1.0))
        w_nlos = np.sqrt(1.0 / (kappa + 1.0))
        combined = w_los * los_vec + w_nlos * nlos
    return scale * combined


def _rician_scalar_from_components(los: complex, kappa: float, scale: float, rng: np.random.Generator) -> complex:
    nlos = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    if kappa <= 0:
        combined = nlos
    else:
        w_los = np.sqrt(kappa / (kappa + 1.0))
        w_nlos = np.sqrt(1.0 / (kappa + 1.0))
        combined = w_los * los + w_nlos * nlos
    return scale * combined


class ChannelGenerator:
    """生成并管理场景信道。

    使用示例：
      gen = ChannelGenerator(config, seed=0)
      gen.reset()  # 随机生成设备位置
      chans = gen.generate_channels()
    """

    def __init__(self, config, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.devices = None

        # 固定几何设置（题设）
        area = float(self.config.channel.area_size)
        # BS (30,0), RIS (30,30), Eve (50,50) 按题设
        self.bs_pos = np.array([30.0, 0.0, 1.0])
        self.ris_pos = np.array([30.0, 30.0, 1.0])
        self.eve_pos = np.array([50.0, 50.0, 1.0])

    def reset(self, seed: Optional[int] = None):
        """随机生成 K 个设备坐标，x,y 在 [0, area_size]，z 取 1.0 m。"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        K = int(self.config.system.K)
        area = float(self.config.channel.area_size)
        xs = self.rng.uniform(0.0, area, size=K)
        ys = self.rng.uniform(0.0, area, size=K)
        zs = np.ones(K)  # 设备高度，简化为 1m
        self.devices = np.column_stack((xs, ys, zs))
        return self.devices

    def _angle_from_ris(self, node_pos: np.ndarray) -> float:
        """计算从 RIS 指向 node 的角度 phi（弧度），相对于 RIS 法线 x 轴正方向。"""
        dy = node_pos[1] - self.ris_pos[1]
        dx = node_pos[0] - self.ris_pos[0]
        phi = np.arctan2(dy, dx)
        return phi

    def generate_channels(self, device_positions: Optional[np.ndarray] = None, return_torch: bool = False) -> Dict[str, np.ndarray]:
        """生成并返回信道字典（按题设要求的键和结构）。

        如果 `return_torch=True`，返回值中的 numpy arrays 会被转换为 torch tensors（dtype=complex64）。
        """
        if device_positions is None:
            if self.devices is None:
                raise ValueError('devices not set: call reset() or pass device_positions')
            devices = self.devices
        else:
            devices = np.array(device_positions, dtype=float)

        K = int(self.config.system.K)
        M = int(self.config.system.M)
        assert devices.shape[0] == K, f'devices length {devices.shape[0]} != K={K}'

        h_k_r: List[np.ndarray] = []
        h_k_b: List[complex] = []
        h_k_e: List[complex] = []

        # RIS -> BS and RIS -> Eve LoS vectors
        phi_rb = self._angle_from_ris(self.bs_pos)
        phi_re = self._angle_from_ris(self.eve_pos)
        a_rb = ula_response(M, phi_rb)
        a_re = ula_response(M, phi_re)

        # distances and scaling
        for k in range(K):
            dev = devices[k]
            # device -> RIS
            d_kr = _safe_distance(self.ris_pos, dev)
            beta_kr = self.config.channel.beta_kr
            eps = self.config.channel.ref_loss
            scale_kr = np.sqrt(eps * (d_kr ** (-beta_kr)))
            kappa_kr = self.config.channel.kappa_rk
            phi_kr = self._angle_from_ris(dev)
            a_kr = ula_response(M, phi_kr)
            # NLoS vector
            nlos_kr = (self.rng.standard_normal(M) + 1j * self.rng.standard_normal(M)) / np.sqrt(2)
            if kappa_kr <= 0:
                hk_r = scale_kr * nlos_kr
            else:
                w_los = np.sqrt(kappa_kr / (kappa_kr + 1.0))
                w_nlos = np.sqrt(1.0 / (kappa_kr + 1.0))
                hk_r = scale_kr * (w_los * a_kr + w_nlos * nlos_kr)
            h_k_r.append(hk_r)

            # device -> BS (direct)
            d_kb = _safe_distance(self.bs_pos, dev)
            beta_kb = self.config.channel.beta_kb
            scale_kb = np.sqrt(eps * (d_kb ** (-beta_kb)))
            kappa_kb = self.config.channel.kappa_kb
            # LoS scalar set to 1
            if kappa_kb <= 0:
                hk_b = scale_kb * (self.rng.standard_normal() + 1j * self.rng.standard_normal()) / np.sqrt(2)
            else:
                w_los = np.sqrt(kappa_kb / (kappa_kb + 1.0))
                w_nlos = np.sqrt(1.0 / (kappa_kb + 1.0))
                hk_b = scale_kb * (w_los * 1.0 + w_nlos * (self.rng.standard_normal() + 1j * self.rng.standard_normal()) / np.sqrt(2))
            h_k_b.append(hk_b)

            # device -> Eve (direct)
            d_ke = _safe_distance(self.eve_pos, dev)
            beta_ke = self.config.channel.beta_ke
            scale_ke = np.sqrt(eps * (d_ke ** (-beta_ke)))
            kappa_ke = self.config.channel.kappa_ke
            if kappa_ke <= 0:
                hk_e = scale_ke * (self.rng.standard_normal() + 1j * self.rng.standard_normal()) / np.sqrt(2)
            else:
                w_los = np.sqrt(kappa_ke / (kappa_ke + 1.0))
                w_nlos = np.sqrt(1.0 / (kappa_ke + 1.0))
                hk_e = scale_ke * (w_los * 1.0 + w_nlos * (self.rng.standard_normal() + 1j * self.rng.standard_normal()) / np.sqrt(2))
            h_k_e.append(hk_e)

        # RIS -> BS (vector)
        d_rb = _safe_distance(self.ris_pos, self.bs_pos)
        beta_rb = self.config.channel.beta_rb
        scale_rb = np.sqrt(eps * (d_rb ** (-beta_rb)))
        kappa_rb = self.config.channel.kappa_rb
        nlos_rb = (self.rng.standard_normal(M) + 1j * self.rng.standard_normal(M)) / np.sqrt(2)
        if kappa_rb <= 0:
            h_r_b = scale_rb * nlos_rb
        else:
            w_los = np.sqrt(kappa_rb / (kappa_rb + 1.0))
            w_nlos = np.sqrt(1.0 / (kappa_rb + 1.0))
            h_r_b = scale_rb * (w_los * a_rb + w_nlos * nlos_rb)

        # RIS -> Eve
        d_re = _safe_distance(self.ris_pos, self.eve_pos)
        beta_re = self.config.channel.beta_re
        scale_re = np.sqrt(eps * (d_re ** (-beta_re)))
        kappa_re = self.config.channel.kappa_re
        nlos_re = (self.rng.standard_normal(M) + 1j * self.rng.standard_normal(M)) / np.sqrt(2)
        if kappa_re <= 0:
            h_r_e = scale_re * nlos_re
        else:
            w_los = np.sqrt(kappa_re / (kappa_re + 1.0))
            w_nlos = np.sqrt(1.0 / (kappa_re + 1.0))
            h_r_e = scale_re * (w_los * a_re + w_nlos * nlos_re)

        # 把列表转换为 numpy arrays，方便矩阵运算
        h_k_r_np = np.vstack(h_k_r)  # shape (K, M)
        h_r_b_np = np.asarray(h_r_b)
        h_r_e_np = np.asarray(h_r_e)
        h_k_b_np = np.asarray(h_k_b)
        h_k_e_np = np.asarray(h_k_e)

        out = {
            'h_k_r': h_k_r_np,
            'h_r_b': h_r_b_np,
            'h_r_e': h_r_e_np,
            'h_k_b': h_k_b_np,
            'h_k_e': h_k_e_np,
        }

        if return_torch:
            try:
                import torch
            except Exception as e:
                raise RuntimeError('torch is required for return_torch=True') from e

            def _to_torch(x: np.ndarray):
                # ensure complex64 to keep memory small and compatibility
                if not isinstance(x, np.ndarray):
                    x = np.array(x)
                x32 = x.astype(np.complex64)
                return torch.from_numpy(x32)

            out_torch = {k: _to_torch(v) for k, v in out.items()}
            return out_torch

        return out


if __name__ == '__main__':
    # 简单自测
    try:
        from new_paper.config import Config
    except Exception:
        import importlib.util
        import os
        cfg_path = os.path.join(os.path.dirname(__file__), 'config.py')
        spec = importlib.util.spec_from_file_location('new_paper.config', cfg_path)
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        Config = cfg_mod.Config

    cfg = Config()
    gen = ChannelGenerator(cfg, seed=0)
    devs = gen.reset(seed=1)
    # numpy outputs
    chans_np = gen.generate_channels(return_torch=False)
    print('devices:', devs)
    print('h_k_r shape:', chans_np['h_k_r'].shape)
    print('h_r_b shape:', chans_np['h_r_b'].shape)
    print('h_r_e shape:', chans_np['h_r_e'].shape)
    print('h_k_b shape:', chans_np['h_k_b'].shape)
    print('h_k_e shape:', chans_np['h_k_e'].shape)

    # torch outputs (if torch available)
    try:
        chans_torch = gen.generate_channels(return_torch=True)
        import torch
        print('torch outputs:')
        print('h_k_r type:', type(chans_torch['h_k_r']), 'shape:', tuple(chans_torch['h_k_r'].shape))
        print('h_r_b type:', type(chans_torch['h_r_b']), 'shape:', tuple(chans_torch['h_r_b'].shape))
    except RuntimeError:
        print('torch not available; skipped torch output test')

