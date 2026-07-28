CAD 项目新增内容记录
====================

维护规则
--------
1. FullPICAI（Dataset141）训练权重统一放在 checkpoints_FullPICAI，不与旧 checkpoints 混用。
2. 后续新增 Python 文件或目录时，在本文件“新增内容”中补一条记录。
3. 后续训练任务的 train_dataset 统一使用 enable_augmentation=False；验证集同样为 False。
4. FullPICAI 任务使用独立 task_name，避免覆盖 Dataset131/旧 PI-CAI 结果。

新增内容
--------
2026-07-21
- dataset/PI-CAI/nnUNet_raw/Dataset141_FullPICAI：完整 PI-CAI 数据集划分，训练 1199 例、验证 300 例。
- SegGland_FullPICAI/：Dataset141 的 FullPICAI 训练入口目录。
  - train_nnUNet_FullPICAI.py
  - train_UNet_FullPICAI.py
  - train_UNETR_FullPICAI.py
  - train_TransUNet_FullPICAI.py
  - train_BMA_FullPICAI.py
  - train_ResGNet_FullPICAI.py
  - train_AWUNet_FullPICAI.py
  - train_WaveFormer_FullPICAI.py
  - train_WaTER_FullPICAI.py
  - infer_WaveFormer_FullPICAI.py：WaveFormer FullPICAI 独立推理入口，默认使用 best_dice_model.pth。
  - infer_ResGNet_FullPICAI.py：ResGNet FullPICAI 独立推理入口，按单通道输出阈值验证。
  - infer_BMA_FullPICAI.py：BMA-Net FullPICAI 独立推理入口，默认使用 best_dice_model.pth。
- checkpoints_FullPICAI/：FullPICAI 独立 checkpoint 目录。
  - SegGland_nnUNet_FullPICAI
  - SegGland_TransUNet_FullPICAI
  - SegGland_WaveFormer_FullPICAI
  - SegGland_WaTER_FullPICAI
- infer/run_logs/SegGland_WaveFormer_FullPICAI.log：WaveFormer 训练日志。
- infer/run_logs/SegGland_WaTER_FullPICAI.log：WaTER 训练日志。
- Options 中新增 FullPICAI 专用配置：nnUNet、TransUNet、WaveFormer、WaTER。
- Options_TransUNet_FullPICAI：40 epoch、每 5 epoch 保存，训练/验证均不增强。
- Options_WaTER_FullPICAI：12 epoch、每 2 epoch 保存，训练/验证均不增强。
- Options_UNet_FullPICAI：普通 UNet 的 Dataset141 配置，200 epoch、每 20 epoch 保存、训练/验证均不增强。
- Options_UNETR_FullPICAI：UNETR 的 Dataset141 配置，200 epoch、每 20 epoch 保存、训练/验证均不增强。
- Options_BMA_FullPICAI：BMA-Net 的 Dataset141 配置，400 epoch、训练/验证均不增强。
- Options_ResGNet_FullPICAI：ResGNet 的 Dataset141 配置，50 epoch、训练/验证均不增强。
- Options_AWUNet_FullPICAI：AW_UNet 的 Dataset141 配置，200 epoch、每 30 epoch 保存、训练/验证均不增强；`--resume True` 自动读取对应 task_name 的 model_latest.pth。
- infer_FullPICAI/：所有 FullPICAI 模型的推理结果和推理日志目录。
  - SegGland_nnUNet_FullPICAI_infer/：nnUNet 在 Dataset141 验证集上的预测、轮廓图和指标。
  - SegGland_nnUNet_FullPICAI_infer.log：nnUNet FullPICAI 推理日志。
- test_prostate_val.py：模型任务名或数据路径包含 FullPICAI 时，自动将结果写入 infer_FullPICAI。
- render_prostate_contours.py：验证第二阶段的多进程轮廓图生成器；test_prostate_val.py 会先完成指标和 JSON，再调用该脚本绘图。

迁移说明
--------
2026-07-21：已将四个 FullPICAI 任务从 checkpoints/ 迁移至 checkpoints_FullPICAI/。
WaveFormer 和 WaTER 从迁移前最近的 model_latest.pth 继续训练。
