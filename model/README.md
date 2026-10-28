# 模型文件

正式部署模型位于`mlp-optimized-20261024/final_all_esc50.npz`，结构92→96→5，共9,413个参数。文件SHA256为`2c74151ca171eff88af7efa26b8a4b3dfc37ea42c7e77b0c9b379bd61610f9e9`。

`mlp-study-20261026/`保存最近一轮十组消融的60个模型，包括五个官方保留折模型和一个全量模型。该目录用于实验复现，与正式部署模型分工明确。

所有训练从随机初始化开始，数据来自ESC-50固定390条子集。权重与数据用途说明见[第三方资源](../THIRD_PARTY_NOTICES.md)。导出命令位于[训练目录](../training/mlp_study/README.md)。
