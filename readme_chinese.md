@"
# RIS-Assisted Over-the-Air Federated Learning: Simulation and Reproduction

（可重构智能表面辅助的空中联邦学习：仿真与复现）

本文件为项目的中文说明，完整翻译自项目根目录下的 README.md，用于帮助中文读者快速理解代码结构、运行方式与依赖环境。

## 项目简介

该仓库为论文 "Reconfigurable intelligent surface enabled federated learning: A unified communication-learning design approach" 的仿真代码实现与结果复现包。论文作者：Hang Liu, Xiaojun Yuan, Ying-Jun Angela Zhang。论文已发表于/投稿至 IEEE Transactions on Wireless Communications（亦可参见 ArXiv 版本：https://arxiv.org/abs/2011.10282）。本代码基于 Python 3 编写，旨在复现论文中提出的联合通信—学习设计方法，并提供可重复运行的实验流程。

## 论文摘要（译）

为利用移动边缘网络中产生的大量数据，联邦学习（Federated Learning, FL）被提出作为集中式机器学习（ML）的替代方案。通过在边缘设备上协同训练共享模型，FL 避免了直接传输原始数据，从而在通信延迟与隐私方面优于集中式 ML。为提升 FL 中模型聚合的通信效率，已引入空中计算（over-the-air computation）以支持大量设备同时上传本地模型，利用无线信道的叠加特性。然而，设备间通信能力的异质性导致空中 FL 面临"拖延者（straggler）"问题：最差信道设备会成为聚合性能的瓶颈。设备选择能在一定程度上缓解该问题，但会引发数据利用与通信可靠性之间的权衡。本文提出利用可重构智能表面（RIS）技术来缓解空中 FL 的拖延者问题，建立通信—学习联合分析框架，并将设备选择、空中收发机设计与 RIS 配置联合优化。数值实验表明所提方法在学习精度上显著优于现有方法，尤其在设备间信道差异较大时。

## 依赖项

- Python >= 3.5
- torch
- torchvision
- scipy
- CUDA（如使用 GPU 加速）

注：若需固定可复现的 Python 环境，建议使用虚拟环境并在运行前安装上述库。

## 使用方法（概览）

主脚本为 main.py，代码在运行时支持通过命令行参数配置仿真实验设置（详见 main.py 中的 initial() 函数）。主要可配置参数如下：

| 参数名 | 含义 | 默认值 | 类型/范围 |
| ------ | ---- | ------ | ---------- |
| M | 设备总数 | 40 | int |
| N | 接收天线总数 | 5 | int |
| L | RIS 元素数 | 40 | int |
| nit | 算法 1（SCA）的最大迭代次数 I_max | 100 | int |
| Jmax | Gibbs 采样迭代次数 | 50 | int |
| threshold | 算法 1 的早停阈值 | 1e-2 | float |
| tau | 算法 1 的 SCA 正则化项 | 1 | float |
| trial | 蒙特卡洛试验次数 | 50 | int |
| SNR | 信噪比 P_0/sigma_n^2（dB） | 90.0 | float |
| verbose | 输出详细程度 | 0 | 0,1,2 |
| set | 使用哪种仿真设置（见论文 V-A 节） | 2 | 1,2 |
| seed | 随机种子 | 1 | int |
| gpu | 用于训练/加速的 GPU 索引（如可用） | 1 | int |
| momentum | SGD 动量（多次本地更新时使用） | 0.9 | float |
| epochs | 训练轮数 T | 500 | int |

示例命令（在命令行/终端运行）：

python -u main.py --gpu=0 --trial=50 --set=2

## 代码结构与主要模块说明

- main.py：仿真入口，解析命令行参数、调用优化与学习流程，并将实验结果存为 store/*.npz。
- optlib.py：实现论文中的优化算法，包括基于 SCA 的局部优化（算法1）和基于 Gibbs 采样的全局搜索（算法2）。主要函数：Gibbs()、find_obj_inner()、sca_fmincon()。
- flow.py：学习流程管理模块，负责读取优化结果、初始化学习模型、执行训练/测试流程（learning_flow()、Learning_iter()、FedAvg_grad()）。
- train_script.py：数据加载、局部更新与模型评估函数（包括 Load_FMNIST_IID()、local_update()、test_model()）。
- AirComp.py：实现空中聚合（over-the-air aggregation）功能，负责在通信层面对本地梯度进行模拟聚合。
- Nets.py：定义用于实验的神经网络结构（如 CNNMnist()）。
- Monte_Carlo_Averaging.py：对 store/ 中保存的多个试验结果进行平均和统计分析。
- data/：存放 Fashion-MNIST 数据集（首次运行时会自动从网络下载）。
- store/：存放仿真输出文件（*.npz）。

## 文档与开发提示

- 仔细查看 main.py 中的 initial() 函数以调整实验参数。
- 建议添加 .gitignore，排除 __pycache__/、*.pyc、store/*.npz、data/ 等不必要或过大的文件。
- 若要复现论文数值结果，请使用论文中指定的参数组合与随机种子，并在尽可能相同的硬件/软件环境下运行。

## 结果保存

- 仿真输出以 .npz 文件保存在 store/ 目录，文件格式与 Monte_Carlo_Averaging.py 的读取逻辑保持一致，便于对多次试验进行统计分析。

## 引用

若使用本代码进行研究并发表，请引用：

Hang Liu, Xiaojun Yuan, and Ying-Jun Angela Zhang. "Reconfigurable intelligent surface enabled federated learning: A unified communication-learning design approach," IEEE Transactions on Wireless Communications, 2020.（ArXiv: https://arxiv.org/abs/2011.10282）

## 许可证

请参考仓库根目录的 LICENSE 文件以获取许可和使用条款。
"@ | Out-File -Encoding utf8 readme_chinese.md


我将定位项目中解析命令行参数位置（main.py 的 initial()）并展示对应默认值与配置位置。

下面简短列出论文中常用的实验/复现设置、代码中对应的位置（可点击打开），以及复现时要保证的一致性与快速运行命令。

代码位置（关键）

参数解析（默认值）: main.py 的 initial() — 查看默认实验参数（M,N,L,trial,SNR,set,seed,nit,Jmax,...）main.py:22-51
运行设备 / 随机种子 / 结果文件名: 主流程（设置 device、np.random.seed、结果保存名）main.py:60-70
数据分布设置（set=1 / set=2 的逻辑）: 设备样本数 K、两类场景的生成（set==1 为均匀分配，else 为两簇＋不均等）main.py:101-116
本地训练与数据划分代码（如何把 K 映射到每个用户的数据）: train_script.py 的 Load_FMNIST_IID() 和 mnist_iid()（数据划分实现）train_script.py:1-80
论文里常用的实验设置（对应代码中的参数）

设备数量 M、接收天线 N、RIS 元素 L：--M, --N, --L（见 initial()）main.py:22-26
蒙特卡洛试验次数：--trial（默认 50）main.py:38-39
信噪比（dB）：--SNR（默认 90.0）main.py:38-39
场景类型：--set（1 = concentrated + equal dataset；2 = two clusters + unequal dataset）main.py:41-42
随机种子：--seed（默认 1）——会用到 np.random.seed(libopt.seed) 保证可重复main.py:60-63
优化迭代与 SCA/Gibbs：--nit, --Jmax, --threshold, --tau（见 initial()）main.py:30-33
学习相关：--local_ep, --local_bs, --lr, --momentum, --epochs（见 initial()），训练在 flow.learning_flow() 中调用main.py:46-51 与 train_script.py:1-80
要完全复现论文结果必须保证的环境一致性

相同的随机种子：通过 --seed 传入并确保在所有随机源（numpy, torch）都设置（当前代码设置了 numpy 随机种子；建议在运行前也设置 torch 的 seed）。
相同的数据划分：libopt.K 在 main.py 决定每设备样本数，Load_FMNIST_IID() 根据该 K 划分数据（确认 --set 对 K 的生成逻辑）main.py:101-116 与 train_script.py:1-80。
相同的软件/硬件：相同 PyTorch 版本、CUDA 版本与相近 GPU；建议记录 torch.__version__ 与 torch.cuda.is_available() 以及 nvidia-smi 输出。
相同超参：--lr, --epochs, --local_ep, --local_bs 等必须与论文的实验表一致（代码里默认 lr=0.01, epochs=500，paper 的设置请对照论文 Section V 的实验描述）。
快速示例（在项目根目录运行，复现论文常用设置）

示例命令（与论文一致的参数）：
若要保证 torch 随机性也固定，建议在运行脚本前在 Python 层设置：
建议你接下来的操作（二选一）

我可以：1) 帮你在代码里再明确设置 torch 的随机种子（在 main.py 调用 np.random.seed 之后加入 torch.manual_seed），并提交一次修改。
或者：2) 我把你需要运行的完整复现命令和检查环境的命令发给你（包括检查 torch 版本、CUDA、GPU），你在本地运行并把输出发给我，我再帮你核对差异。
你想先让我（A）在 main.py 中补上 torch 随机种子的修改并提交，还是（B）我把检查与运行命令直接发给你？