# 版本指南

本项目采用两个独立文件夹管理：

## 公开发布版（public version）
- 路径：`stock-quant-rl-course-design-public`
- 用途：GitHub 公开发布
- 内容：代码、轻量文档、必要说明
- 不包含：完整数据、模型、输出文件

## 本地完整开发版（local full version）
- 路径：`stock-quant-rl-course-design`
- 用途：本地开发、实验、完整结果
- 内容：所有代码、数据、模型、输出文件

## 设计原则
1. public 版不包含所有 outputs 是有意设计，不是实验缺失
2. 完整结果可通过运行 `run_stage1.py` 到 `run_stage6.py` 重新生成
3. 最终报告 Markdown 位于 `docs/大数据课设报告_最终版.md`
4. docx 报告仅保留在本地 full 版

## 同步策略
- 从 full 版同步到 public 版时，只同步轻量文件
- 不同步：data/、outputs/models/、完整 outputs/tables/、完整 outputs/figures/
- 同步：README.md、FINAL_PROJECT_CHECK.md、docs/*.md、outputs/EXPERIMENT_SUMMARY.md