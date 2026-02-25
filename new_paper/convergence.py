"""
Convergence constraint checker based on Wu et al. (2025) Theorem 1 and constraint (17c).

提供函数：
- `check_convergence_constraint(x, K, epsilon=0.001, mu=1.0, F_diff=1.0, Omega=100, delta=1.0)` -> (satisfied: bool, bound: float)
- `compute_convergence_penalty(x, K, ...)` -> float penalty (0 if satisfied, large number otherwise)

包含简单测试用例。
"""
from typing import Sequence, Tuple


def check_convergence_constraint(x: Sequence[int],
                                 K: int,
                                 epsilon: float = 0.001,
                                 mu: float = 1.0,
                                 F_diff: float = 1.0,
                                 Omega: int = 100,
                                 delta: float = 1.0) -> Tuple[bool, float]:
    """检查参与者选择 x 是否满足收敛精度约束。

    参数:
      x: 长度为 K 的二值序列（0/1），表示是否选择设备
      K: 设备总数
      epsilon: 收敛阈值（线性值，例如 0.001 对应 -30 dB）
      mu, F_diff, Omega, delta: 公式参数（见论文与注释）

    返回:
      (satisfied, bound)
        satisfied: True 如果上界 <= epsilon
        bound: 计算得到的上界值

    计算公式（按题设与论文）:
      Term1 = (2 * mu * F_diff) / (Omega + 1)
      Term2 = 2 * delta
      A = sum(x)
      Term3 = (2 * delta * (K - 2*A)) / (A)   if A>0
      bound = Term1 + Term2 + Term3

    注意: 若 A == 0 返回 False 且 bound 为很大值。
    """
    # 校验输入长度
    if len(x) != K:
        raise ValueError(f'Length of x ({len(x)}) must equal K ({K})')

    # 将 x 转为整数并计算 A
    A = int(sum(int(bool(xi)) for xi in x))

    if A <= 0:
        # 没有参与者，约束无法满足
        bound = float('inf')
        return False, bound

    Term1 = (2.0 * mu * F_diff) / float(Omega + 1)
    Term2 = 2.0 * delta
    Term3 = (2.0 * delta * (K - 2.0 * A)) / float(A)

    total_bound = Term1 + Term2 + Term3
    satisfied = (total_bound <= float(epsilon))
    return satisfied, float(total_bound)


def compute_convergence_penalty(x: Sequence[int],
                                K: int,
                                epsilon: float = 0.001,
                                mu: float = 1.0,
                                F_diff: float = 1.0,
                                Omega: int = 100,
                                delta: float = 1.0,
                                penalty_value: float = 100.0) -> float:
    """当不满足约束时返回惩罚值，否则返回 0.

    penalty_value: 当不满足约束时返回的罚分，可用于奖励/惩罚设计。
    """
    sat, bound = check_convergence_constraint(x, K, epsilon, mu, F_diff, Omega, delta)
    return 0.0 if sat else float(penalty_value)


if __name__ == '__main__':
    # 简单测试用例
    K = 5
    examples = {
        'x_2': [1, 1, 0, 0, 0],  # A=2
        'x_5': [1, 1, 1, 1, 1],  # A=5
        'x_0': [0, 0, 0, 0, 0],  # A=0
    }

    for name, x in examples.items():
        sat, bound = check_convergence_constraint(x, K, epsilon=0.001, mu=1.0, F_diff=1.0, Omega=100, delta=1.0)
        penalty = compute_convergence_penalty(x, K, epsilon=0.001)
        print(f'{name}: A={sum(x)} -> bound={bound:.6g}, sat={sat}, penalty={penalty}')
