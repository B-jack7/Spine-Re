# 许可范围与代码来源

本仓库包含新增维护代码、历史模型实现和影像数据，不能将全部内容视为同一许可证授权。根目录 [LICENSE](../LICENSE) 的 MIT 授权范围如下。

## MIT 授权的维护者原创部分

仅对 B-jack7 有权授权的原创贡献适用：

- `reproduce/audit.py`
- `reproduce/metrics.py`
- `reproduce/train.py`
- `reproduce/test_core.py`
- `reproduce/requirements.txt`
- `scripts/check_notebooks.py`
- `scripts/notebook_process.py`
- `tests/test_notebook_process.py`
- `.github/workflows/checks.yml`
- 根目录 `README.md`、`reproduce/README.md`、`docs/RESULTS.md` 和本文的原创说明文字。

这些维护代码在 2026 年 9 月的维护提交中加入。MIT 允许使用、修改、分发和商业使用，并要求保留版权和许可声明；不提供担保。第三方依赖仍适用各自许可证。

`reproduce/train.py` 会加载下面所述的历史模型。它自身的原创部分获 MIT 授权，不代表包含历史模型的完整训练程序可以仅按 MIT 分发；组合程序仍需满足适用的上游许可条件。

## 本次 MIT 授权不覆盖的内容

- `J-Unet网络结构部分代码及注释说明/` 下全部历史代码、注释、数据和其他文件。
- 所有 MRI 图像、分割标签、预测图、模型权重、检查点、实验产物，以及 `docs/assets/` 中的图片。
- 两个 Colab 笔记本整体（`notebooks/Spine_Re_Quickstart.ipynb`、`reproduce/Spine_Re_Colab.ipynb`）及其下载的内容。笔记本中上述 MIT 文件的逐字副本继续适用该文件的 MIT 授权，但这不扩大到笔记本其他内容或下载的模型、数据。
- 其他未列入上述授权清单的文件及第三方贡献。

原有权利和许可不因本次说明而改变。本次增加许可证不补授历史材料的再分发权限，也不能证明数据来源或影像使用权限已经核实。

## 2026-10-06 来源检查记录

检查基于提交 `0166bf2150244ec6cf7bbe1ca403a8b549770b97`。

- 当时仓库没有 LICENSE、COPYING 或第三方许可清单。
- 历史 `UNET/unet/unet_parts.py` 的类结构、说明文字和两条修复链接与 [milesial/Pytorch-UNet 的对应文件](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py) 高度相似；该上游目前提供 [GPL-3.0 许可文本](https://github.com/milesial/Pytorch-UNet/blob/master/LICENSE)。这是一条需要追溯的来源线索，不是对本仓库实际复制版本或完整授权链的确认。此文件及相关派生内容不在本次 MIT 授权范围内。
- 历史评估代码 `evalution_segmentaion.py` 含有指向 `shelhamer/model.berkeleyvision.org` 的注释链接，也需要核对具体来源和许可。
- 原有 README 已说明 PNG 数据来源及标签准确性未得到独立核实；本次未确认影像、标签或预测图的再分发许可。

后续需要由维护者补充实际使用的上游仓库与版本，核实相应许可并保留原始声明，同时记录数据来源及授权依据。在这些事项解决前，请勿将整个仓库描述为“全部 MIT 授权”，或仅凭此文件声称完整项目的开源资格已经核实。
