# 正式 Tiny MLP 模型

`final_all_esc50.npz`为当前唯一部署模型，使用ESC-50固定390条五类子集从随机初始化训练60轮，结构为92→96→5、9,413个参数。时域输入进行asinh压缩，再与MFCC/Delta特征共同标准化。

`deployment.json`记录训练配置、模型SHA256及C头文件SHA256；`sha256.json`给出当前NPZ的校验值。C权重位于`app/audiodetect/mlp_weights.h`，采用INT8存储及浮点运算。

固定配方五折337/390（86.41%）的折模型保存在`../mlp-study-20261026/full/`。外部65条开发集分类为57/65（87.69%）；五类统一计分，该集合参与过项目开发选择。
