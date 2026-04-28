# 课程展示说明

## 当前项目情况

这是一个手势识别方向的计算机视觉课程实践。当前默认主线已经切到浏览器中文展示端，同时保留 Python/CV 本地系统控制模式。整体工程仍然包含数据集读取、模型训练、评估、单图推理、摄像头实时手势展示、性能测试和展示资产生成。

核心分类模型已经从 LeNet 基线升级为改进 CNN，输入为 `28x28` 灰度手势图像，输出为 Sign Language MNIST 的 24 个静态字母类别。`J` 和 `Z` 需要动态轨迹，因此不属于当前静态分类任务。现场摄像头展示使用 MediaPipe 手部关键点做稳定实时手势识别：浏览器中文端负责展示和交互答辩，Python/OpenCV 备用线负责真实鼠标/键盘控制；CNN 字母分类保留为训练评估成果。

## 可以讲的亮点

- 端到端闭环完整：从 CSV 数据集到模型权重、指标、图表、摄像头脚本和演示视频。
- 有模型升级对比：LeNet 基线测试准确率 `83.81%`，改进 CNN 测试准确率 `99.93%`。
- 不只展示准确率：新增逐类别 F1、易混类别和弱类别分析。
- 有可复现实验资产：`outputs/improved_train` 和 `outputs/improved_eval` 保存训练曲线、权重、指标和混淆矩阵。
- 有答辩素材：`build_showcase.py` 会生成课程展示看板、样本宫格和中文报告。
- 有实时应用入口：浏览器中文端可实时展示单手准备、单手捏合拖动、双手缩放旋转和双手张开重置；Python 备用线仍可识别 `Open Palm`、`Point`、`Pinch`、`Victory`、`Fist` 并安全开启真实鼠标/键盘控制；字母模式可裁剪 ROI 后做 24 类分类。

## 当前结果

- LeNet 基线测试准确率：`83.81%`
- 改进 CNN 测试准确率：`99.93%`
- 改进 CNN Macro F1：`0.999`
- 改进 CNN Weighted F1：`0.999`
- 改进 CNN 最佳验证准确率：`100.00%`
- 验证到测试泛化差距：`0.07%`
- 摄像头无窗口 10 秒测试：`252` 帧，平均 `27.38 FPS`

答辩时建议强调两层能力：离线分类模型通过结构升级和轻量数据增强显著提升准确率；在线演示通过 MediaPipe 关键点保证实时稳定，避免把静态字母分类硬说成动态手语翻译。

## 展示顺序建议

1. 先展示 `outputs/showcase/sample_grid.png`，说明任务是 24 类静态手势识别。
2. 展示 `outputs/showcase/course_showcase.png`，给出整体指标和训练趋势。
3. 展示 `outputs/improved_eval/confusion_matrix.png`，说明改进后只剩很少混淆。
4. 展示 `outputs/showcase/course_report.md` 中的弱类别分析，体现误差分析能力。
5. 最后先运行 `./run_camera_demo.sh`，展示中文看板、实时骨架、状态指标和 3D 交互；如果答辩老师更关心真实桌面控制，再补跑 `./run_showcase_demo.sh`，说明安全锁定后按 `c` 开启真实控制。

## 常用命令

```bash
cd /home/xzr/桌面/cv-work/gesture_lenet
source ~/miniconda3/bin/activate gesture-py310
python evaluate.py --checkpoint outputs/final_train/best_model.pth --test-csv data/raw/sign_mnist_test.csv --output-dir outputs/final_eval
python evaluate.py --checkpoint outputs/improved_train/best_model.pth --test-csv data/raw/sign_mnist_test.csv --output-dir outputs/improved_eval
python build_showcase.py
python camera_benchmark.py --camera /dev/video0 --frames 120
python infer_camera.py --mode showcase --camera /dev/video0 --control-mode off --process-fps 20 --duration 20 --window-mode off --save-video outputs/demo/showcase.mp4 --save-metrics outputs/demo/showcase_metrics.json
```
