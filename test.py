from ultralytics import YOLO

"""
测试自定义模型
"""

if __name__ == '__main__':
    model = YOLO("runs/detect/train4/weights/best.pt")
    # 推理并保存图片
    results = model("example/3.png", save=True)
    print('推理成功😄')

