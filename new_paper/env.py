"""
Simple RL environment for RIS-assisted FL that integrates convergence and secrecy penalties.

Environment API (minimal):
- `env = RLFLEnv(config)`
- `obs = env.reset()`
- `obs, reward, done, info = env.step(action)`

Action expected (for this minimal version): a dict with keys:
 - 'x': binary array length K (0/1) for participant selection
 - 'b_k': array length K of bandwidth fractions (non-negative, ideally sum<=1)
 - 'theta': array length M of RIS phase angles in radians OR complex diagonal entries exp(1j*theta)

Reward composition (discrete penalties by default):
 reward = - max_delay - p_bandwidth - p_convergence - p_secrecy

Note: This is a prototype to integrate penalties; replace `compute_max_delay` with real delay model later.
"""
from typing import Tuple, Dict, Any
import numpy as np

try:
    from new_paper.convergence import compute_convergence_penalty, check_convergence_constraint
    from new_paper.secrecy import compute_secrecy_rates
    from new_paper.channel import ChannelGenerator
except Exception:
    # fallback when running file directly
    import importlib.util
    import os
    base = os.path.dirname(__file__)
    spec_c = importlib.util.spec_from_file_location('new_paper.convergence', os.path.join(base, 'convergence.py'))
    mod_c = importlib.util.module_from_spec(spec_c)
    spec_c.loader.exec_module(mod_c)
    compute_convergence_penalty = mod_c.compute_convergence_penalty
    check_convergence_constraint = mod_c.check_convergence_constraint

    spec_s = importlib.util.spec_from_file_location('new_paper.secrecy', os.path.join(base, 'secrecy.py'))
    mod_s = importlib.util.module_from_spec(spec_s)
    spec_s.loader.exec_module(mod_s)
    compute_secrecy_rates = mod_s.compute_secrecy_rates

    spec_ch = importlib.util.spec_from_file_location('new_paper.channel', os.path.join(base, 'channel.py'))
    mod_ch = importlib.util.module_from_spec(spec_ch)
    spec_ch.loader.exec_module(mod_ch)
    ChannelGenerator = mod_ch.ChannelGenerator


class RLFLEnv:
    def __init__(self, config, seed: int = 0):
        self.config = config
        self.K = int(config.system.K)
        self.M = int(config.system.M)
        self.gen = ChannelGenerator(config, seed=seed)
        self.rng = np.random.default_rng(seed)

        # penalty magnitudes (tunable)
        # lowered to be comparable with delay (seconds) per user request
        # typical values: bandwidth 0.1, secrecy 0.1, convergence 0.01
        self.penalty_bandwidth = 0.1
        self.penalty_secrecy = 0.1
        # reduce convergence penalty default by one order
        self.penalty_convergence = 0.001
        # expose a minimum bandwidth fraction that can be adapted at runtime
        self.min_frac = 0.01

    def reset(self, seed: int = None) -> Dict[str, Any]:
        if seed is not None:
            self.gen.reset(seed=seed)
            self.rng = np.random.default_rng(seed)
        else:
            self.gen.reset()
        # initial observation can be channel or positions; keep simple
        chans = self.gen.generate_channels(return_torch=False)
        obs = {'channels': chans}
        return obs

    def compute_max_delay(self, x: np.ndarray, b_k: np.ndarray) -> float:
        """根据论文公式计算实际延迟：T_loc + T_tr（每个参与者），返回最大延迟（秒）。

        T_loc = c_k * D_k / f_k
        T_tr = S_k / R_kb
        使用 compute_secrecy_rates 获得 R_kb（到 BS 的速率）和 R_ks（保密速率）。
        注意：compute_secrecy_rates 返回的 rates_info 是字典 keyed by participant index.
        """
        participants = [i for i, xi in enumerate(x) if int(xi) == 1]
        if len(participants) == 0:
            # no participants selected -> no transmission delay
            self.last_delays = {}
            return 0.0

        # ensure theta and channels are set in env; using self.theta/self.channels if present
        try:
            theta = self.theta
            channels = self.channels
        except AttributeError:
            # fallback: regenerate channels and use default theta (all ones)
            channels = self.gen.generate_channels(return_torch=False)
            theta = np.ones(self.M, dtype=np.complex128)

        rates_info, _ = compute_secrecy_rates(theta, participants, [i for i in range(self.K) if i not in participants], channels, b_k, self.config, backend='numpy')

        c_k = float(self.config.fl.c_k) if hasattr(self.config, 'fl') and hasattr(self.config.fl, 'c_k') else float(getattr(self.config, 'c_k', 1000))
        D_k = float(self.config.fl.D_k) if hasattr(self.config, 'fl') and hasattr(self.config.fl, 'D_k') else float(getattr(self.config, 'D_k', 6250))
        f_k = float(self.config.fl.f_k) if hasattr(self.config, 'fl') and hasattr(self.config.fl, 'f_k') else float(getattr(self.config, 'f_k', 1e9))
        S_k = float(self.config.fl.S_k_bits) if hasattr(self.config, 'fl') and hasattr(self.config.fl, 'S_k_bits') else float(getattr(self.config, 'S_k', 3e6))

        T_loc = (c_k * D_k) / f_k
        # minimum allowed rate to avoid divide-by-zero (bps)
        # increase to avoid huge transmission times when rates are extremely small
        min_rate = 1e-3
        # maximum transmission time cap (seconds) to keep rewards finite
        # lower the cap to 1e3s to prevent single steps dominating episode reward
        max_T_tr = 1e3

        delays = {}
        for k in participants:
            info_k = rates_info.get(k, None)
            if info_k is None:
                # missing rates -> treat as very large delay but finite
                delays[k] = float(1e8)
                continue
            R_kb = info_k.get('R_kb', 0.0)
            # guard against non-finite or zero rates which would produce infinite delay
            try:
                R_kb_val = float(R_kb)
            except Exception:
                R_kb_val = 0.0
            if not np.isfinite(R_kb_val) or R_kb_val <= min_rate:
                T_tr = min(S_k / min_rate, max_T_tr)
            else:
                T_tr = S_k / R_kb_val
                if T_tr > max_T_tr:
                    T_tr = max_T_tr
            delays[k] = T_loc + T_tr

        # return maximum delay among participants
        max_delay = max(delays.values()) if delays else float('inf')
        # store for info
        self.last_delays = delays
        return float(max_delay)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """执行一步环境交互。

        action: dict with keys 'x', 'b_k', 'theta'
        returns: obs, reward, done, info
        """
        # parse action
        x = np.asarray(action.get('x'))
        b_k = np.asarray(action.get('b_k'))
        theta = np.asarray(action.get('theta'))

        if x.shape[0] != self.K:
            raise ValueError('action x must have length K')
        if b_k.shape[0] != self.K:
            raise ValueError('action b_k must have length K')
        if theta.shape[0] != self.M:
            raise ValueError('action theta must have length M')

        # normalize theta if given as angles
        if np.isrealobj(theta):
            theta_complex = np.exp(1j * theta)
        else:
            theta_complex = theta.astype(np.complex128)

        # generate channels for current positions and store
        chans = self.gen.generate_channels(return_torch=False)
        self.channels = chans
        self.theta = theta_complex

        # p_bandwidth: penalty if sum(b_k) > 1
        sum_b = float(np.sum(b_k))
        p_bandwidth = 0.0
        if sum_b > 1.0 + 1e-12:
            p_bandwidth = self.penalty_bandwidth

        # p_convergence via compute_convergence_penalty (continuous form)
        # compute_convergence_penalty returns a penalty value (discrete). To build a continuous penalty,
        # call check_convergence_constraint to get the bound and scale penalty by bound/epsilon - 1.
        # use epsilon from config if available
        eps = 0.001
        if hasattr(self.config, 'fl') and hasattr(self.config.fl, 'epsilon_conv'):
            try:
                eps = float(self.config.fl.epsilon_conv)
            except Exception:
                eps = 0.001

        _ = compute_convergence_penalty(list(map(int, x)), self.K, epsilon=eps, penalty_value=self.penalty_convergence)
        sat_flag, bound_val = check_convergence_constraint(list(map(int, x)), self.K, epsilon=eps)
        # continuous convergence penalty: use log compression to reduce growth
        # keep an upper cap to avoid extreme values
        max_conv_penalty = 10.0
        if bound_val == float('inf'):
            p_convergence = max_conv_penalty
        else:
            ratio = max(1.0, bound_val / eps)
            # log-compressed penalty
            p_convergence = self.penalty_convergence * float(np.log(ratio))
            # ensure non-negative
            p_convergence = max(0.0, p_convergence)
            # cap
            if p_convergence > max_conv_penalty:
                p_convergence = max_conv_penalty

        # p_secrecy: compute detailed rates and use continuous penalty per participant
        b_k_fracs = b_k.astype(float)
        participant_indices = [i for i, xi in enumerate(x) if int(xi) == 1]
        jammer_indices = [i for i in range(self.K) if int(x[i]) == 0]

        rates_info, sats = compute_secrecy_rates(theta_complex, participant_indices, jammer_indices, chans, b_k_fracs, self.config, backend='numpy')
        p_secrecy = 0.0
        # sats aligns with participant_indices; also rates_info keyed by participant index
        for idx, sat in zip(participant_indices, sats):
            if not sat:
                # continuous secrecy penalty based on deficit fraction
                Rks = rates_info[idx]['R_ks']
                Rmin = float(self.config.fl.R_min)
                deficit = max(0.0, (Rmin - Rks) / (Rmin + 1e-12))
                p_secrecy += deficit * self.penalty_secrecy

        # compute delays using realistic formula
        max_delay = self.compute_max_delay(x, b_k_fracs)

        # 对数奖励：log1p 对小数敏感，对大数压缩
        # 将 convergence/secrecy 惩罚再缩小一阶以避免与延迟惩罚竞争
        reward = -np.log1p(max_delay) - p_bandwidth - p_convergence / 10.0 - p_secrecy / 10.0
        # 说明：log1p(0)=0, log1p(1)=0.69, log1p(10)=2.40, log1p(100)=4.62

        obs = {'channels': chans, 'rates_info': rates_info}
        info = {'p_bandwidth': p_bandwidth, 'p_convergence': p_convergence, 'p_secrecy': p_secrecy, 'max_delay': max_delay, 'bound_conv': bound_val}
        done = False
        return obs, reward, done, info


if __name__ == '__main__':
    # 简单自测环境
    try:
        from new_paper.config import Config
    except Exception:
        import importlib.util, os
        base = os.path.dirname(__file__)
        spec = importlib.util.spec_from_file_location('new_paper.config', os.path.join(base, 'config.py'))
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        Config = cfg_mod.Config

    cfg = Config()
    env = RLFLEnv(cfg, seed=0)
    obs = env.reset(seed=1)
    K = cfg.system.K
    M = cfg.system.M

    # construct a random action
    x = np.array([1,1,0,0,0])
    b_k = np.ones(K) / float(K)
    theta = np.zeros(M)  # all-zero phases -> ones

    action = {'x': x, 'b_k': b_k, 'theta': theta}
    obs, reward, done, info = env.step(action)
    print('reward:', reward)
    print('info:', info)
