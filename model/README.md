# 项目模型

当前默认为 `mlp-retrained-20261024/`，保留完整五折与独立全量模型。以下为历史v11原模型说明。最终部署模型与各折训练模型不同；五折指标不是最终模型在训练集上重新评分得到的。

`tiny_mlp_mfcc92_robust_simulator.npz` 是保留的历史v11模型：

- 架构：92→64→5
- 参数：6,277
- SHA256：`c96333aaae66d324b8b8cec48ffb0ec981ac4e975d3cf9a7d4a2fdba4307c97e`

模型采用泄漏安全的混合增强训练（时间平移、音量缩放、时间拉伸、高斯噪声，并混合同训练折真实背景），
五折指标见 `results/python_cv/metrics.json`。

开发板与主机 C 模拟器使用的 C 权重头文件全部由 `training/export_to_c.py` 从该模型生成：

- `app/audiodetect/model_weights.h`（float32）
- `app/audiodetect/model_weights_int8.h`（weight-only INT8）
- `simulator/generated/model_weights_int8.h`（weight-only INT8）

## 来源与使用边界

模型由项目流程在ESC-50选择集上训练，不使用第三方预训练声学模型。原数据不随仓库发布；数据集整体许可为CC BY-NC 3.0，单片段条款和署名由上游列明。

权重供研究原型复现。源码Apache-2.0许可不替代数据使用条件，也不保证权重可无条件商业部署。实际商业用途、再分发和共同权利人授权应另核对；本说明不推定权重当然继承数据许可。详见`THIRD_PARTY_NOTICES.md`。
