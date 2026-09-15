# 训练与评估入口

项目介绍、数据格式、指标说明及 Colab 操作步骤见 [项目 README](../README.md)。

[在 Colab 打开快速运行笔记本](https://colab.research.google.com/github/B-jack7/Spine-Re/blob/main/notebooks/Spine_Re_Quickstart.ipynb)

```bash
python reproduce/test_core.py
python reproduce/audit.py --output runs/audit
python reproduce/train.py --output runs/smoke --epochs 2 --train-limit 8 --val-limit 4 --test-limit 4
python reproduce/train.py --output runs/junet_50 --epochs 50 --pause-every 15
python reproduce/train.py --output runs/junet_50 --epochs 50 --pause-every 15 --resume
```

小实验和正式实验使用不同输出目录。续跑必须保留相同代码、模型、种子、batch size、学习率及标签策略。默认忽略非 0/100/255 标签颜色；该选择会影响评估结果。GPU 默认使用混合精度。测试集仅在计划总轮数结束后评估。

`data_audit.json` 和 `manifest.csv` 记录数据检查结果，包括旧预测文件的单独重算；它们不是当前训练过程的指标日志。已完成运行的指标请查看 [RESULTS](../docs/RESULTS.md)。
