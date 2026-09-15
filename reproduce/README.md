# Spine-Re 可审计的重新实验

本目录为原项目补充独立训练入口。**当前版本不能被称为论文的精确复现，也不能证明原始数据真实性。** 原项目代码、PNG 和历史结果保留用于比较。

## 已核实的复现差异

参考论文：Zhou et al., *Enhancing spinal MRI segmentation with an asymmetric U-Net architecture*, DOI: [10.36922/aih.3889](https://doi.org/10.36922/aih.3889)，表 1–4、图 2 和图 4。

| 项目 | 论文 | 原仓库实际内容 |
|---|---|---|
| 数据 | 215 位患者的 T2 NIfTI；4:1 划分 | 2523 对 PNG；train 1764 / val 253 / test 506，约 7:1:2 |
| 患者信息 | 患者队列 | PNG 仅有连续数字文件名，无患者映射，不能核实患者间独立性 |
| 标签 | 0 / 100 / 255 | 约 1.6%–1.7% 像素出现其他颜色；旧编码器静默归背景 |
| 损失 / batch / epochs | Dice / 2 / 100 | CE / 1 / 60；图 4 横轴又画到 300，训练预算存在歧义 |
| 解码器通道 | 120 | 180 |
| J-Unet 参数 | 22,331,979 | 实例化实测 25,659,999 |
| Accuracy | 像素准确率公式 | 训练脚本记录背景和椎体的平均召回率，排除了椎间盘 |
| 训练 / 验证 | 每轮各自运行 | 旧代码仅在循环前调用 train()，首轮验证后留在 eval() |
| 权重选择 | 未完整说明 | 旧代码按训练 mIoU 保存；仓库没有可加载权重 |

以上差异说明现有材料不足以精确复现论文，不能据此判断成因或认定造假。没有发现像素内容完全相同的跨集合图像；该检查不识别同一患者的相邻切片或近似重复。

## 使用

在 Google Colab 上传或打开 `Spine_Re_Colab.ipynb`，选择 T4 GPU。笔记本包含固定原始提交的下载、代码安装、指标测试、数据审计、训练和结果导出。

命令行从仓库根目录运行（依赖 Colab 已安装的 PyTorch、NumPy、Pillow）：

```bash
python reproduce/test_core.py
python reproduce/audit.py --output runs/audit
python reproduce/train.py --output runs/junet_100 --epochs 100 --model junet
python reproduce/train.py --output runs/unet_100 --epochs 100 --model unet
```

续跑原实验，除 epochs 外保持配置、代码和输出路径相同：

```bash
python reproduce/train.py --output runs/junet_100 --epochs 100 --model junet --resume
```

GPU 默认开启混合精度，以降低显存占用；`--no-amp` 使用 FP32。默认真实 batch size 为 2，不用梯度累积冒充同一 BatchNorm 行为。默认 512×512 输入，标签不缩放。只按需读取当前 batch，不把全数据集载入内存。Adam 学习率 1e-4，固定随机种子；设备、库版本、参数量、代码哈希、配置、逐轮验证、逐图测试和混淆矩阵均落盘。相同种子不保证跨硬件逐位相同。

最佳权重依据验证集前景 Dice 选择，测试集仅在本次训练结束后评估。不要根据测试结果选择超参数；需要更改方法时，应另设独立测试集。缺少患者映射时，本仓库测试只能称为“给定切片划分上的测试”。

## 标签和指标：必须连同结果一起说明

- `--label-policy ignore`（默认）：仅精确匹配 0 / 100 / 255，其余像素在损失和评估中忽略。这不是恢复原始标注；忽略模糊边界会改变评估任务，可能使分数提高。
- `--label-policy legacy-background`：复现旧编码器，把其他颜色归背景，用于敏感性分析，不代表正确标注。
- `dice_per_class` 顺序为背景、椎体、椎间盘；`dice_macro_foreground` 为后两类均值。
- 主汇总来自所有有效测试像素组成的全局混淆矩阵；`image_macro_averages` 为逐图计算再平均。二者不能混用。
- 分母为零的类别记作 null，均值忽略这些类别。仅 `legacy_mean_recall_background_vertebra` 保留旧代码的无真值类别记零规则。
- 所有预测都输出 PNG，可与配套标签重新计算。以全局前景指标为主，同时保留背景指标，避免背景占比掩盖错误。

## 历史预测图重算（不是重新训练结果）

原始提交 `b678a3130c7f2472208f33ce9c129ecc297061b0` 的 506 张 `J-Unet/result_pics`：

| 规则 / 汇总 | Dice | mIoU | 像素准确率 |
|---|---:|---:|---:|
| 旧标签规则，逐图均值，包含背景 | 0.872985 | 0.810579 | 0.983731 |
| 旧标签规则，全局，包含背景 | 0.916860 | 0.851231 | 0.983731 |
| 忽略未知标签，逐图均值，包含背景 | 0.892868 | 0.843566 | 0.988394 |
| 忽略未知标签，全局，包含背景 | 0.938305 | 0.886624 | 0.988406 |

论文表 3 的 J-Unet 为 Dice 0.8913、Accuracy 0.9729、mIoU 0.8288。现有预测图与这些数字不能直接对应。历史预测图的生成权重、训练配置和过程缺失，其结果只用于文件审计。

## 文件说明

- `data_audit.json` / `manifest.csv`：审计输出；哈希、像素计数、数据划分与旧图重算。
- `config.json`：运行配置及环境。
- `history.jsonl`：每轮训练损失与验证指标。
- `best.pt`：验证集最佳模型；`last.pt`：完整优化器、AMP 与随机状态，用于续跑。只加载自己生成的可信 checkpoint。
- `test_metrics.json` / `test_per_image.json` / `predictions/`：本次真实运行输出。

Colab 的 `/content` 会随运行时释放而丢失。及时下载结果 ZIP；长期训练可在笔记本里选择挂载个人 Drive 保存 checkpoint。免费 GPU 可用性和最长运行时受 Colab 限制，100 轮可能需要多次续跑。没有自动付费或保活绕过逻辑。
