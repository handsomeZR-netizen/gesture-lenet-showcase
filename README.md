# Gesture Recognition Course Project

这个目录现在更适合作为“计算机视觉课程实践”来展示：它有静态手势识别模型训练、测试集评估、单图推理、摄像头实时手势控制计算机、性能测试和课程展示资产生成。浏览器 `web_control_demo` 仍然保留，但当前默认主线是 Python/OpenCV 本地展示界面。

## 课程展示主线

推荐答辩时把主线讲成：

```text
Sign Language MNIST CSV
-> 数据加载与标签映射
-> LeNet 基线与改进 CNN 训练
-> 测试集评估与混淆矩阵
-> 单图/摄像头推理
-> 展示看板与误差分析
```

当前已有结果：

- 改进模型权重：`outputs/improved_train/best_model.pth`
- 改进模型训练曲线：`outputs/improved_train/training_curves.png`
- LeNet 基线测试准确率：`83.81%`
- 改进 CNN 测试准确率：`99.93%`
- 改进 CNN Macro F1：`0.999`
- 改进 CNN Weighted F1：`0.999`
- 改进 CNN 混淆矩阵：`outputs/improved_eval/confusion_matrix.png`
- Python 手势控制界面：`infer_camera.py --mode showcase --control-mode mouse`
- 摄像头演示指标：`outputs/demo/showcase_metrics.json`
- 课程展示包：`outputs/showcase/`

一键重新生成评估与展示资产：

```bash
cd /home/xzr/桌面/cv-work/gesture_lenet
source ~/miniconda3/bin/activate gesture-py310

python evaluate.py \
  --checkpoint outputs/improved_train/best_model.pth \
  --test-csv data/raw/sign_mnist_test.csv \
  --output-dir outputs/improved_eval

python build_showcase.py
```

生成的展示文件包括：

- `outputs/showcase/course_showcase.png`：可直接放 PPT 的总览看板
- `outputs/showcase/course_report.md`：中文课程展示报告
- `outputs/showcase/per_class_f1.png`：逐类别 F1 分析
- `outputs/showcase/sample_grid.png`：24 类样本宫格
- `outputs/improved_eval/per_class_metrics.csv`：逐类别指标表
- `outputs/improved_eval/confusion_pairs.json`：主要易混类别

现场摄像头展示：

```bash
cd /home/xzr/桌面/cv-work/gesture_lenet
./run_showcase_demo.sh
```

无窗口录制和测试：

```bash
python camera_benchmark.py --camera /dev/video0 --frames 120
python infer_camera.py \
  --mode showcase \
  --camera /dev/video0 \
  --control-mode off \
  --process-fps 20 \
  --duration 20 \
  --window-mode off \
  --save-video outputs/demo/showcase.mp4 \
  --save-metrics outputs/demo/showcase_metrics.json
```

## 保留的演示线

这个目录仍然包含两条演示线：

- `python_control_showcase`
  - 当前默认主线
  - 使用 OpenCV 摄像头 + `MediaPipe Hand Landmarker` + 手部关键点规则
  - 在一个 1280x720 的教学看板里展示摄像头、手势、性能和控制状态
  - 可选真实控制鼠标/键盘：`Point` 移动、`Pinch` 点击/拖拽、`Victory` 切换窗口、`Fist` 退出/重置
- `gesture_control_demo`
  - 保留浏览器版 3D 球体控制演示
  - 入口是 `./run_web_control_demo.sh`
- `legacy_letter_demo`
  - 保留旧版字母分类演示
  - 使用 `sign-language-mnist` 重新训练出来的 LeNet 风格分类模型
  - 入口仍然是 `infer_camera.py` / `infer_image.py` / `train.py`

## 当前主线

默认启动的不是字母分类，也不是 Web 页面，而是 Python 手势控制计算机展示界面。

它的控制逻辑是：

- `Open Palm`：准备状态，不触发系统动作
- `Point`：移动鼠标指针
- `Pinch`：短按点击，长按拖拽
- `Victory`：触发 `Alt+Tab`
- `Fist`：触发 `Esc` / 重置控制状态

为避免误触，程序启动后默认是安全锁定状态，需要在 OpenCV 窗口中按 `c` 才会开始真实控制；按 `space` 可以暂停/恢复，按 `q` 或 `Esc` 退出。当前控制模式用的是“关键点 + 规则”，不是旧的字母分类模型。

## 为什么和 PDF 里的字母图不一样

PDF 里的那张图本质上是“静态测试样本可视化”。而当前项目先前训练的是公开 `sign-language-mnist` 数据集的字母分类器：

- 它只能在那套数据集的标签空间里做静态分类
- 它不是为真实摄像头连续交互设计的
- 数据集标签本身和 PDF 里的示意内容也不完全一致

所以现在把旧模型明确降级为 `legacy`，避免继续把它和新版控制交互混在一起解释。

## 目录说明

```text
gesture_lenet/
├── web_control_demo/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── run_camera_demo.sh
├── run_showcase_demo.sh
├── run_web_control_demo.sh
├── run_legacy_letter_demo.sh
├── infer_camera.py
├── infer_image.py
├── train.py
├── evaluate.py
├── camera_benchmark.py
├── build_showcase.py
├── download_dataset.py
├── check_environment.py
├── models/
│   └── hand_landmarker.task
├── outputs/
└── docs/
```

## 快速启动

Python 手势控制计算机演示：

```bash
cd /home/xzr/桌面/cv-work/gesture_lenet
./run_camera_demo.sh
```

这个命令会启动本地 OpenCV 展示窗口。窗口内按键：

- `c`：解除安全锁，开启真实鼠标/键盘控制
- `space`：暂停/恢复控制
- `q` 或 `Esc`：退出

浏览器 3D 球体演示仍可手动启动：

```bash
./run_web_control_demo.sh
```

旧版字母分类演示：

```bash
source ~/miniconda3/bin/activate gesture-py310
./run_legacy_letter_demo.sh
```

## 旧版字母模型

旧版字母模型文件仍然保留在：

```text
outputs/final_train/best_model.pth
```

它是你之前这套工程里已经训练好的 LeNet 基线模型。新的高准确率字母模型在：

```text
outputs/improved_train/best_model.pth
```

`run_legacy_letter_demo.sh` 现在默认使用改进 CNN 权重；旧 LeNet 权重仍保留用于对比实验。

## 自定义模型训练现状

当前已经实现的是：

- 默认控制模式：`MediaPipe Hands + 规则`
- 旧版对照模式：`LeNet + sign-language-mnist`
- 改进分类模式：`Improved CNN + BatchNorm/Dropout + 轻量数据增强`

当前还没有落地的是“你自己的控制手势数据集训练”。这一步需要先采你自己的控制语义数据，再训练新的离散命令模型。现在项目已经把默认主线切到不依赖旧字母模型的控制模式，后续可以在这个基础上继续补“自己的模型”。

## 文档

- [QuickStart](docs/QuickStart.md)
- [课程展示说明](docs/CourseShowcase.md)
- [控制模式与模型说明](docs/控制模式与模型说明.md)
- [前端需求文档](docs/前端需求文档.md)
