# 多模态算法

让模型"看懂"图像：从把图像切块编码（ViT），到把视觉特征提取、映射并对齐进语言空间（跨模态投影与融合）。

> 📌 返回 [项目主 README](../README.md) ｜ 总览见主页 [模块导航](../README.md#modules)

## 文件清单

| 文件 | 你将学到 | 关键步骤 |
| --- | --- | --- |
| [`vit.ipynb`](./vit.ipynb) | Vision Transformer 全流程 | Patch Embedding → 位置编码 → Transformer Encoder → 完整 ViT → 分块可视化 |
| [`image_feature_extraction.ipynb`](./image_feature_extraction.ipynb) | 图像特征的提取与映射 | 特征提取器 → 特征映射模块 → 完整流水线 → 特征可视化 → 端到端示例 |
| [`projection_layer.ipynb`](./projection_layer.ipynb) | 跨模态投影与对齐 | 投影层设计 → 跨模态融合层 → 对比学习 → 端到端训练 |

## 推荐学习路径

1. **`vit`**：先理解图像如何变成 patch 序列、被 Transformer 处理；
2. **`image_feature_extraction`**：理解视觉特征的提取与映射；
3. **`projection_layer`**：理解视觉特征如何投影 / 对齐到语言表征空间。

## 依赖

```bash
pip install numpy torch matplotlib jupyter
```
