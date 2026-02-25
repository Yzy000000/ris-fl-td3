"""
Configuration for reproducing the paper (Sec. V-A parameters).

组织：将参数分为系统参数（SystemParams）、信道参数（ChannelParams）、联邦学习参数（FLParams）。
注：注释中标明来源为论文 Sec. V-A。单位在注释或变量名中已转换为 SI 基本单位（比特、Hz、W、cycles）。
"""
from dataclasses import dataclass


@dataclass
class SystemParams:
    """系统参数（来自论文 Sec. V-A）"""
    K: int = 5  # 设备总数
    M: int = 50  # RIS 单元数
    P_k: float = 0.1  # 发射功率，单位：W（参与者和干扰者相同）


@dataclass
class ChannelParams:
    """信道参数（来自论文 Sec. V-A）"""
    # 噪声功率，单位：W
    sigma2_b: float = 1e-14
    sigma2_e: float = 1e-14

    # 路径损耗指数
    beta_kr: float = 2.2
    beta_rb: float = 2.2
    beta_re: float = 2.2
    beta_kb: float = 3.6
    beta_ke: float = 3.6

    # 莱斯因子（Rician K-factor）
    kappa_rk: float = 4.0
    kappa_rb: float = 4.0
    kappa_re: float = 4.0
    kappa_kb: float = 0.0
    kappa_ke: float = 0.0

    # 参考路径损耗（假设在 1m 处的损耗）
    ref_loss: float = 1e-3

    # 区域大小（正方形边长，单位：m）
    area_size: float = 60.0

    # 带宽，默认 10 MHz（论文 Fig.3(c) 中带宽范围 5-15 MHz，取中点）
    B: float = 10e6


@dataclass
class FLParams:
    """联邦学习与计算/通信相关参数（来自论文 Sec. V-A）"""
    # 模型参数大小 S_k = 3 Mbit -> 转换为比特
    S_k_bits: int = 3 * 10**6  # bits

    # CPU 频率 f_k = 1 GHz
    f_k: float = 1e9  # Hz

    # CPU 周期/样本 c_k = 1000 cycles/bit
    c_k: int = 1000  # cycles/bit

    # 本地数据量 D_k = 6250 bits
    D_k: int = 6250  # bits

    # 最低保密速率 R_min = 2e4 bps
    R_min: float = 2e4  # bps

    # 收敛精度阈值 epsilon_conv = -30 dB -> 线性 0.001
    epsilon_conv: float = 0.001

    # 每个训练回合的通信轮次 Omega（用于收敛约束）
    Omega: int = 100


class Config:
    """总配置类，按组暴露参数对象。可直接 `from new_paper.config import Config` 并访问 `Config.system` 等属性。"""

    system: SystemParams = SystemParams()
    channel: ChannelParams = ChannelParams()
    fl: FLParams = FLParams()

    @classmethod
    def as_dict(cls):
        """返回配置字典（便于日志记录/序列化）。"""
        return {
            "system": cls.system.__dict__,
            "channel": cls.channel.__dict__,
            "fl": cls.fl.__dict__,
        }


if __name__ == "__main__":
    # 简单自测：打印配置，便于快速检查
    cfg = Config()
    import json

    print(json.dumps(cfg.as_dict(), indent=2))
