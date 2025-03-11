from ultralytics import YOLO


if __name__ == '__main__':

    # 加载 YOLOv8 预训练模型（如果有专门的海洋垃圾模型，替换 "yolov8n.pt"）
    model = YOLO("yolov8n.pt")

    for name in model.names:
        print(model.names[name])

    # 读取图像
    image_path = "./example/2.png"
    # image = cv2.imread(image_path)

    # 进行推理
    results = model(image_path,save=True)
