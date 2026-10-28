# Tiny MLP 板级实验工具

`verify_results.py`复核已发布真板数据的正确数、每类召回、FPR、告警门限、时延分布、输入及模型哈希，并核对采集格式、短时循环和故障记录。

```bash
python3 scripts/board_benchmark_mlp/verify_results.py
```

命令在仓库根目录执行，无需开发板。对应的实际板测程序保存在`results/board-mlp-20261027/source-evidence/`，测量结果见[实验说明](../../results/board-mlp-20261027/README.md)。
