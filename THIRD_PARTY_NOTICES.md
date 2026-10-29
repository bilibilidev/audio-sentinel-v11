# 第三方资源与署名

Audio Sentinel的应用源码在根目录LICENSE规定的范围内发布。下列数据、库、系统和工具分别适用其原始许可，源码许可与数据、模型及厂商二进制的条款分别处理。

|资源|项目用途|来源与许可说明|
|---|---|---|
|ESC-50，Karol J. Piczak|390条五类子集训练与五折评测|[项目](https://github.com/karolpiczak/ESC-50)，数据集整体CC BY-NC 3.0，逐片段署名随上游记录|
|FSD50K|外部65条开发集的音频来源之一|[官方项目](https://github.com/eduardofv/FSD50K)，逐音频来源及许可记录于`results/mlp-study-20261026/external-sources.json`|
|Donate-a-Cry|外部65条开发集的音频来源之一|数据来源与哈希保留在实验输入清单；仓库未附可验证的逐条许可记录，原始录音不在本仓库分发|
|PANNs Cnn6|AudioSet预训练冻结特征基线|[官方实现](https://github.com/qiuqiangkong/audioset_tagging_cnn)、[官方权重](https://zenodo.org/records/3987831)，版本及哈希见`results/mlp-study-20261026/panns-provenance.json`，原始checkpoint和上游代码另行获取|
|OpenVela / NuttX、NuttX Apps|设备系统与应用框架|[OpenVela](https://github.com/open-vela)，依各项目原许可和通知使用，实际依赖见板测resolved manifest|
|LVGL|小屏图形界面|MIT；板级补丁附许可原文`board/r528s3-gemini-s1/patches/LVGL-LICENSE.txt`|
|Allwinner / Rivotek BSP|R528启动、音频、显示与触摸|厂商源码和二进制依各自条款提供，本仓库发布应用配置、启动脚本及最小补丁|
|Python、NumPy、PyTorch、SciPy、scikit-learn、librosa等|CPU训练、音频处理与实验|依各项目许可使用；实际实验版本见训练和基线协议，不随仓库打包运行环境|
|GCC与交叉工具链|主机和板级构建|工具及运行库依其发行包条款使用，实际版本随构建记录保存|

## 数据、模型与模板

ESC-50由Karol J. Piczak发布，数据集许可及片段署名见[上游LICENSE](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE)。训练增强包含时间平移、音量缩放、时间拉伸、噪声与同折背景混合，原素材条款继续适用。复现数据由使用者按来源单独获取。

正式Tiny MLP由ESC-50子集从随机初始化训练，未使用第三方预训练权重。PANNs只用于独立精度基线，具有额外AudioSet预训练来源。外部65条为项目开发集，原始音频、joblib模型及预训练checkpoint不在仓库分发。

关键词模板由`training/keyword_detect.py`的项目合成器生成，并导出为`app/audiodetect/keyword_templates.h`；它们用于MFCC+DTW命令行原型，测试分数描述合成样例识别。

## 来源与使用范围

项目早期应用来源保留在[NOTICE](NOTICE)中。当前发布包含后续Tiny MLP训练、C集成、板级适配与实验。公开构建记录同时保存resolved manifest、工具链和本地补丁，描述相应镜像的依赖组成。

数据和模型的商业用途、原始录音再分发及厂商固件交付，应按实际资源条款办理。AI工具曾用于编码、排错和文档整理，设备端分类使用本地Tiny MLP，AI服务不是运行时依赖。
