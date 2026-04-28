# QuickStart

如果目标是课程答辩或项目展示，优先跑第 0 步生成展示包，再跑第 1 步现场手势控制计算机演示。浏览器控制演示仍可运行，但 Python/CV 主线已经是默认入口。

## 0. 生成课程展示包

```bash
cd /home/xzr/桌面/cv-work/gesture_lenet
source ~/miniconda3/bin/activate gesture-py310

python evaluate.py \
  --checkpoint outputs/improved_train/best_model.pth \
  --test-csv data/raw/sign_mnist_test.csv \
  --output-dir outputs/improved_eval

python build_showcase.py
```

重点查看：

```text
outputs/showcase/course_showcase.png
outputs/showcase/course_report.md
outputs/showcase/per_class_f1.png
outputs/showcase/sample_grid.png
outputs/improved_eval/confusion_pairs.json
outputs/improved_eval/per_class_metrics.csv
```

## 1. 启动 Python 手势控制计算机展示

```bash
cd /home/xzr/桌面/cv-work/gesture_lenet
./run_showcase_demo.sh
```

这个模式会显示实时手部骨架、规则手势、FPS、延迟、控制状态和教学说明。启动后默认安全锁定，在窗口中按 `c` 才会开始真实控制鼠标/键盘；按 `space` 暂停/恢复，按 `q` 或 `Esc` 退出。

控制手势：

- `Open Palm`：准备状态
- `Point`：移动鼠标指针
- `Pinch`：短按点击，长按拖拽
- `Victory`：`Alt+Tab`
- `Fist`：`Esc` / 重置

无窗口录制测试：

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

## 2. 启动 Web 手势控制演示

当前项目也保留浏览器版 `gesture_control_demo`。


```bash
cd /home/xzr/桌面/cv-work/gesture_lenet
./run_web_control_demo.sh
```

脚本会：

- 在本地启动一个静态服务
- 默认地址是 `http://127.0.0.1:8765/web_control_demo/`
- 自动尝试在 Ubuntu 里打开浏览器

进入页面后：

- 点击 `Start Camera`
- 允许浏览器访问摄像头
- 把手放到镜头前开始控制 3D 球体

控制方式：

- 单手张开：准备状态
- 单手捏合：拖动球体
- 双手同时捏合：缩放并旋转球体
- 双手同时张开并保持约 `1.2s`：重置球体

## 3. 如果要跑字母分类摄像头演示

```bash
cd /home/xzr/桌面/cv-work/gesture_lenet
source ~/miniconda3/bin/activate gesture-py310
./run_legacy_letter_demo.sh
```

这个模式会使用：

```text
outputs/improved_train/best_model.pth
```

说明：

- 它使用改进 CNN 权重
- 它依赖 MediaPipe 先检测手，再裁剪 ROI 做 24 类字母分类
- 它适合和训练指标一起讲，不建议把它包装成动态手语翻译

## 4. 环境检查

如果你要继续使用 Python 版训练或字母推理，再运行：

```bash
source ~/miniconda3/bin/activate gesture-py310
python check_environment.py
```

## 5. 重新训练改进字母模型

```bash
source ~/miniconda3/bin/activate gesture-py310
python train.py \
  --train-csv data/raw/sign_mnist_train.csv \
  --test-csv data/raw/sign_mnist_test.csv \
  --output-dir outputs/improved_train \
  --epochs 12 \
  --batch-size 256 \
  --architecture improved \
  --augment \
  --lr 0.001 \
  --weight-decay 0.0005
```

## 6. 评估改进字母模型

```bash
source ~/miniconda3/bin/activate gesture-py310
python evaluate.py \
  --checkpoint outputs/improved_train/best_model.pth \
  --test-csv data/raw/sign_mnist_test.csv \
  --output-dir outputs/improved_eval
```

## 7. 单图字母分类

```bash
source ~/miniconda3/bin/activate gesture-py310
python infer_image.py \
  --checkpoint outputs/improved_train/best_model.pth \
  --image /path/to/your/image.jpg \
  --output-json outputs/predict/result.json \
  --save-preview outputs/predict/preview.png
```

## 8. 最常用文件

Python 控制主线：

```text
infer_camera.py
run_showcase_demo.sh
models/hand_landmarker.task
```

Web 保留演示：

```text
web_control_demo/index.html
web_control_demo/app.js
```

改进字母分类：

```text
outputs/improved_train/best_model.pth
outputs/improved_eval/metrics.json
outputs/improved_eval/confusion_matrix.png
outputs/demo/showcase_metrics.json
```
