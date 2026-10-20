# R528S3 Gemini S1 板级配置

当前 `r528s3-gemini-s1/configs/nsh/defconfig` 为 2026-10-20 真板验证的 128 MiB DDR + 板载 ILI9341 320×240 小屏配置。RAM 从 `0x41400000` 到 `0x47fe7000`，避开顶部 100 KiB RAMLOG；关闭 framebuffer/externinit 以进入物理 LCD 注册分支，使用 `/dev/lcd0`。`AE_AUDIODETECT_UI_AUTOSTART` 启用，rcS 启动 `audiodetect ui`。

`patches/lvgl-aligned-buffers.patch` 是此构建必需的 LVGL 对齐及释放适配，随 overlay 幂等检查应用。原 `reference/openvela_active.config` 与 `reference/nuttx_savedefconfig` 保留为历史 v11 冻结参考，不会随新版配置覆盖；不再与当前 defconfig 字节相同。

在隔离且来源明确的完整 OpenVela SDK 中：

```sh
sh audio-sentinel-v11/scripts/apply_board_overlay.sh /absolute/path/to/sdk
# 正式镜像构建与归档使用：
bash audio-sentinel-v11/scripts/build_r528s3_cnn_image.sh /absolute/path/to/sdk /absolute/path/to/fresh-output
```

构建入口检查隔离标记、应用提交和模型验收记录，不安装依赖、不烧录。不要套用此配置到 goldfish 或 7 寸 MIPI 屏。具体应用路径须与 manifest 映射一致。

最新实际 manifest、工具链、补丁、镜像哈希和板级边界见 [真板结果](../results/board-20261020/README.md)。用户分区擦除烧录后当前小屏/自启正常、WAV 推理通过；保留数据升级、软件重启和实际采集稳定性不能视为已解决。旧镜像与备份保留，不宣称重建镜像与历史镜像字节一致。
