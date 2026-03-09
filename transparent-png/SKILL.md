---
name: transparent-png
description: 将用户上传的图片背景去掉，输出透明背景的 PNG。适用于棋盘格/双色背景：先从边缘像素聚类得到两种背景色，再用 ImageMagick 的连通 floodfill 将背景变为透明。
allowed-tools: ["exec", "read", "write"]
---

# Transparent PNG

把一张图片的背景去掉，得到透明背景的 PNG。

## 适用场景

- 背景是**棋盘格/双色块**（常见于 UI 截图、素材图）
- 主体内部存在与背景接近的灰色：需要**连通域约束**，避免误删主体

## 依赖

- ImageMagick 7（推荐）：`magick` 命令可用
- Python 3
- Pillow（Debian/Ubuntu 推荐用 apt 安装）：`sudo apt-get install -y python3-pil`

## 用法

```bash
# 默认 fuzz=25
bash $SKILL_DIR/run.sh -i input.png -o output.png

# 手动调 fuzz
bash $SKILL_DIR/run.sh -i input.png -o output.png --fuzz 20
bash $SKILL_DIR/run.sh -i input.png -o output.png --fuzz 30
```

## 调参建议（fuzz）

- 背景没去干净：把 `--fuzz` 调大（例如 25 → 30/35）
- 主体边缘被吃掉：把 `--fuzz` 调小（例如 25 → 20/15）

## 工作原理（简述）

1) `scripts/bg2colors.py` 从图像四边采样像素，K=2 聚类得到两种背景代表色
2) `run.sh` 调用 `magick`：对 8 个边界种子点分别用两种颜色做 `alpha floodfill`，只把**连通背景**置为透明
