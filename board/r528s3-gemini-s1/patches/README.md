# LVGL 小屏缓冲适配

`lvgl-aligned-buffers.patch`修改`src/drivers/nuttx/lv_nuttx_lcd.c`，按`LV_DRAW_BUF_ALIGN=64`分配显示缓冲，并保存原始分配指针用于释放。对应LVGL提交为`9465c48a115dcd6f00f643b7fadae81e1da2f9d1`，父提交为`0f2a49f588505a00e8b46e25a34581c87291a62a`。

`scripts/apply_board_overlay.sh`在R528 SDK的LVGL检出中应用补丁；已应用时通过反向检查识别。构建记录同时保存实际LVGL HEAD与本地差异。该补丁服务于R528 LCD驱动，goldfish framebuffer另用对应配置。许可原文保存在`LVGL-LICENSE.txt`，镜像依赖记录见`results/board-mlp-20261027/manifest-resolved.xml`。
