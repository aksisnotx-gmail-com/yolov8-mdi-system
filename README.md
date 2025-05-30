# 海洋垃圾检测系统 (Marine Debris Detection System)

基于YOLOv8的海洋垃圾检测与分析系统，使用Streamlit构建Web界面，支持图像、视频和实时摄像头检测。

## 功能特点

- **多媒体检测**：支持图片和视频文件上传，进行垃圾目标检测
- **实时检测**：支持通过摄像头进行实时垃圾检测
- **可视化分析**：以表格、条形图和饼图形式展示检测结果
- **检测日志**：记录每次检测的时间、目标类别和数量
- **数据导出**：支持将检测结果导出为CSV或JSON格式
- **数据集管理**：支持上传自定义数据集进行模型训练和评估

## 系统要求

- Python 3.9+
- CUDA支持（推荐，用于GPU加速）

## 安装步骤

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行应用：

```bash
cd server
streamlit run app.py
```

## 使用说明

### 上传检测

1. 点击"选择文件"上传图片或视频
2. 调整置信度阈值和NMS参数
3. 点击"开始检测"按钮
4. 查看检测结果和统计信息
5. 可选择导出结果为CSV或JSON格式

### 实时检测

1. 选择摄像头ID（默认为0）
2. 调整置信度阈值和NMS参数
3. 点击"开始实时检测"按钮
4. 在弹出的窗口中查看实时检测结果
5. 按ESC键退出检测

### 检测日志

1. 查看所有检测记录的时间、文件名和检测结果
2. 查看检测趋势图和累计统计信息
3. 可选择导出统计数据

### 数据集管理

1. 上传YOLO格式的数据集配置文件(yaml)
2. 设置训练参数（轮数、批次大小、图像大小）
3. 点击"开始训练"按钮进行模型训练
4. 上传评估数据集并点击"开始评估"按钮评估模型性能
5. 查看模型评估指标（mAP、Precision、Recall等）

## 目录结构

```
├── datasets  数据集
│   ├── global.solution.v4i.yolov8
│   ├── normal
│   └── small
├── deploy-linux.sh  Linux 部署脚本
├── example  测试样例
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── test-video.mp4
├── requirements.txt 
├── server
│   ├── app.py 启动文件
│   ├── cfg.py
│   ├── inference.py
│   ├── ui.py
│   └── utils.py
├── test-opencv.py 测试 opencv
├── val.py  验证模型
├── train.py 训练模型
└── yolov8n.pt yolov8n模型
```

## 注意事项

- 首次运行时，系统会自动下载YOLOv8预训练模型
- 对于大型视频文件，处理可能需要较长时间
- 实时检测功能需要摄像头支持

## 许可证

Apache License
