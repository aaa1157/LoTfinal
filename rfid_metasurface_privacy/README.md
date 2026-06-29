# 面向 RFID 无线感知隐私保护的可学习超表面防护仿真与深度攻击评估

## 1. 项目背景

RFID/Wi-Fi 等物联网无线设备不仅能通信，也可能通过无线信道变化感知人体存在、运动、呼吸等隐私信息。本项目基于 RFNOID 论文的思想，通过 1-bit 可编程超表面扰乱 RFID 隔墙人体运动感知信号，从而保护运动隐私，并进一步加入深度学习攻击者和可学习超表面控制器。

**本项目是机理级仿真，不是真实 RFID 硬件复现。**

## 2. 参考论文说明

| 论文 | 作用 |
|------|------|
| RFNOID: Protecting RFID Motion Privacy via Metasurface | 提供 RFID 隔墙人体运动隐私泄露、1-bit 超表面防护、相位混淆、控制目标和评价指标 |
| IRShield: A Countermeasure Against Adversarial Physical-Layer Wireless Sensing | 说明 IRS/超表面可以用于无线感知隐私防护 |
| SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing | 说明深度学习已广泛用于无线人体感知，因此深度学习攻击者是合理的 |
| MetaSensing: Intelligent Metasurface Assisted RF 3D Sensing by Deep Reinforcement Learning | 说明超表面控制可以用学习方法优化 |
| RF-Pose: Through-Wall Human Pose Estimation Using Radio Signals | 说明 RF 信号可以用于隔墙人体感知，隐私风险是真实存在的 |

## 3. 本项目和 RFNOID 的关系

- RFNOID 提出了使用 1-bit 可编程超表面保护 RFID 运动隐私的核心思想
- 本项目借鉴了 RFNOID 的超表面防护思路，但不复刻真实硬件
- 本项目进一步加入了深度学习攻击者（PhaseCNN、PhaseNet-Lite、TinyTCN）
- 本项目进一步加入了可学习超表面控制器（LMC）
- 本项目构建了完整的仿真数据集 RFID-MetaPrivacy-Sim

## 4. 仿真模型

### 复数基带信号模型

```
R(t) = v_static + v_human(t) + v_meta(t) + noise
```

- `v_static`：静态路径，包括 LoS 和环境静态反射
- `v_human(t)`：人体运动或呼吸造成的动态反射
- `v_meta(t)`：超表面反射
- `noise`：复高斯噪声

### 攻击者观测量

```
phi(t) = unwrap(angle(R(t)))
```

### 仿真参数

| 参数 | 值 |
|------|-----|
| 采样率 | 30 Hz |
| 仿真时长 | 60 s |
| 序列长度 | 1800 |
| 超表面单元数 | 16 (4×4) |
| 人体运动频段 | 0.5-2.5 Hz |
| 呼吸频段 | 0.2-0.5 Hz |

**注意：以上参数是为了方法理解和课程实验而设置的简化参数。**

## 5. RFID-MetaPrivacy-Sim 数据集构建方法

- 由机理级 RFID 信道仿真模型生成的合成数据集
- 每条样本包含一段 RFID 接收相位序列和完整 metadata
- 随机化参数包括：静态路径相位/幅度、人体反射幅度/频率/初相、噪声强度、SNR、墙体类型、场景 ID 等
- 严格划分 train/val/test (70%/15%/15%)
- 避免数据泄漏：训练集和测试集不共享完全相同的随机种子

**注意：合成数据集不能替代真实无线感知数据集。**

## 6. 手工超表面策略

| 策略 | 描述 | 目的 |
|------|------|------|
| no_metasurface | v_meta(t) = 0 | 基线，展示隐私泄露 |
| periodic | 所有单元以固定频率翻转 | 展示固定周期扰动容易被滤除 |
| random | 每个时间点随机选择单元翻转 | 提高相位熵 |
| rfnoid_like | RFNOID 启发式策略 | 高相位熵、高时域方差、低频覆盖 |
| multifreq_proposed | 多频子阵列策略 | 增强人体运动频段覆盖，提高滤波难度 |

## 7. 传统攻击者

| 攻击 | 方法 | 输出 |
|------|------|------|
| 方差阈值检测 | mean(var_no_motion) + 3σ | accuracy, TPR, FPR |
| 统计特征分类器 | 12维特征 + Logistic Regression | accuracy, TPR, FPR, F1 |
| 呼吸频率估计 | FFT 在 0.2-0.5 Hz 找峰值 | 估计频率, 绝对误差 |
| 周期干扰滤除分析 | Notch 滤波前后对比 | 滤波前后准确率变化 |

## 8. 深度学习攻击者

| 模型 | 架构 | 参数量 |
|------|------|--------|
| PhaseCNN | 1D-CNN baseline | ~1.9M |
| PhaseNet-Lite | Depthwise separable 1D conv | ~464K |
| TinyTCN | 时序卷积网络 (dilation) | - |

### 深度攻击实验

- **实验 A**：同策略训练测试（验证深度攻击者能识别人体运动）
- **实验 B**：跨策略泛化（测试超表面是否破坏攻击者泛化能力）
- **实验 C**：混合策略强攻击者（模拟更强攻击者适应防护后的识别能力）

## 9. 可学习超表面控制器

### LMC: Learnable Metasurface Controller

- 不直接输出每个时刻每个单元的 0/1 状态
- 输出 multifreq_proposed 的控制参数（子阵列频率、跳频概率、随机翻转概率等）
- 使用随机搜索优化参数
- 损失函数：`loss = attacker_acc - α * phase_entropy - β * lf_ratio + γ * switching_cost`

## 10. 环境配置方法

```bash
bash setup_env.sh
source .venv/bin/activate
```

## 11. 运行方法

### 快速调试

```bash
python build_dataset.py --mode debug --force
python main.py --mode debug
python train_deep_attacker.py --mode debug
python train_learnable_controller.py --mode debug
python generate_report_assets.py
```

### 中等实验

```bash
python build_dataset.py --mode medium --split scene_disjoint --force
python main.py --mode medium
python train_deep_attacker.py --mode medium --seeds 2026 2027 2028
python train_learnable_controller.py --mode medium
```

### 完整实验

```bash
python build_dataset.py --mode full --split scene_disjoint --force
python main.py --mode full
python train_deep_attacker.py --mode full --seeds 2026 2027 2028
python train_learnable_controller.py --mode full
```

### 一键脚本

```bash
# 前台 debug
bash scripts/run_debug.sh

# 后台 medium (SSH 断开不杀进程)
bash scripts/run_medium_nohup.sh

# 后台 full
bash scripts/run_full_nohup.sh
```

## 11.5 SSH 服务器后台运行

在 SSH 服务器上运行 medium/full 实验时，训练可能需要较长时间。使用 nohup 后台运行可以防止 SSH 断开导致进程被杀。

### 前台 debug 测试

```bash
bash scripts/run_debug.sh
```

### 后台运行 medium 实验

```bash
bash scripts/run_medium_nohup.sh
```

### 后台运行 full 实验

```bash
bash scripts/run_full_nohup.sh
```

### 查看状态

```bash
bash scripts/check_status.sh
```

### 实时查看日志

```bash
tail -f results/logs/medium_run_*.log
# 或
tail -f results/logs/full_run_*.log
```

### 停止实验

```bash
bash scripts/kill_experiment.sh medium
# 或
bash scripts/kill_experiment.sh full
```

### tmux 方式（可选）

如果服务器安装了 tmux，也可以使用 tmux：

```bash
tmux new -s rfid_exp
source .venv/bin/activate
python train_deep_attacker.py --mode medium --seeds 2026 2027 2028
```

断开 tmux：`Ctrl+B`，然后按 `D`

重新进入：`tmux attach -t rfid_exp`

**说明**：即使 SSH 断开，只要服务器没有关机、没有重启、没有被云平台回收，nohup 后台进程仍会继续运行。

## 12. 输出文件说明

### 数据文件

| 文件 | 说明 |
|------|------|
| data/processed/rfid_metaprivacy_sim_fast.npz | 快速模式数据集 |
| data/processed/rfid_metaprivacy_metadata_fast.csv | 数据集 metadata |
| data/splits/train_ids.csv | 训练集 ID |
| data/splits/val_ids.csv | 验证集 ID |
| data/splits/test_ids.csv | 测试集 ID |

### 结果表格

| 文件 | 说明 |
|------|------|
| results/tables/metrics.csv | 传统攻击指标 |
| results/tables/respiration_errors.csv | 呼吸估计误差 |
| results/tables/dataset_summary.csv | 数据集统计 |
| results/tables/deep_attack_results.csv | 深度攻击结果 |
| results/tables/controller_results.csv | 控制器对比结果 |

### 图表

| 文件 | 说明 |
|------|------|
| phase_no_metasurface_motion_vs_nomotion.png | 无超表面时运动/静止相位对比 |
| phase_strategies_comparison.png | 各策略相位时域对比 |
| spectrum_strategies_comparison.png | 各策略频谱对比 |
| walking_detection_metrics.png | 运动检测指标对比 |
| respiration_error_comparison.png | 呼吸估计误差对比 |
| summary_comparison.png | 综合对比 |
| dataset_distribution.png | 数据集分布 |
| feature_tsne_or_pca.png | PCA 特征分布 |
| deep_attacker_comparison.png | 深度攻击对比 |
| deep_attacker_train_curves.png | 训练曲线 |
| controller_comparison.png | 控制器对比 |
| controller_search_curve.png | LMC 搜索曲线 |

## 13. 实验结果解释

### RQ1：无超表面时隐私泄露

- 无超表面时，统计分类器准确率 96.5%，TPR 98%
- 呼吸频率估计误差仅 0.022 Hz
- **结论：RFID 接收相位确实泄露人体运动和呼吸隐私**

### RQ2：超表面防护效果

- rfnoid_like 策略将分类器准确率降至 56%，TPR 降至 61%
- multifreq_proposed 策略将分类器准确率降至 58%
- 呼吸估计误差增加到 0.045-0.067 Hz
- **结论：1-bit 可编程超表面能有效降低传统攻击者检测能力**

### RQ3：深度学习攻击者

- 深度学习攻击者在无超表面时准确率 76-88%
- 跨策略泛化实验中，rfnoid_like 下准确率降至 50-65%
- 混合训练后 rfnoid_like 下准确率 65-68%
- **结论：深度攻击者比传统攻击者更强，但仍受超表面影响**

### RQ4：可学习控制器

- LMC 通过随机搜索找到降低攻击者置信度的参数
- 搜索中最优参数将攻击者准确率降至 30%
- **结论：可学习控制器有潜力比人工规则更有效，但需要更高效的优化方法**

## 14. 局限性

- **本项目是机理级仿真，不是真实 RFID 硬件复现**
- 不包含真实 RFID reader、tag、天线、PIN 二极管超表面
- 不包含 CST/HFSS 电磁仿真
- 实验结果不能直接等价于真实硬件性能
- 参数是为了方法理解和课程实验而设置的简化参数
- 合成数据集不能替代真实无线感知数据集
- 深度学习结果依赖仿真数据分布，需要谨慎解释
- LMC 搜索效率有限，更高级的优化方法（如强化学习）可能效果更好

## 15. 常见问题 FAQ

**Q: 为什么不使用真实 RFID 硬件？**
A: 本项目是课程研究项目，目标是理解机理和方法，而非硬件复现。仿真可以灵活调整参数，便于系统性地评估不同策略。

**Q: 为什么深度攻击者准确率不是 100%？**
A: 仿真中 motion 和 no_motion 信号有随机化参数，且经过 detrend 和 zero-mean 预处理，分类器不能只靠平均基线分类。

**Q: 为什么 LMC 控制器效果不如预期？**
A: 当前使用随机搜索，效率有限。更高级的优化方法（如贝叶斯优化、强化学习）可能效果更好。

**Q: 如何在无 GPU 环境运行？**
A: 基础仿真 (main.py) 不依赖 GPU。深度学习脚本会自动使用 CPU，但速度较慢。使用 --fast 模式可加速。
