import time
from time import sleep

import cv2
from ultralytics import YOLO

"""
使用 opencv 弹出一个新的窗口用于展示摄像头捕捉到的画面并使用 yolov8推理并把结果推理到窗口中，我的电脑是 MacBook air m2系列，摄像头 id是 1 依赖我已经下载好了，不用重复下载
所有的代码现在 detected_camera 函数中
"""
def open_camera():
    cap = cv2.VideoCapture(1)  # 选择摄像头 ID（如果 ID=1 不行，尝试 ID=0）

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()  # 读取帧
        if not ret:
            print("无法读取摄像头画面")
            break

        cv2.imshow("Camera Feed", frame)  # 显示摄像头画面

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭窗口



def detected_camera():
    # 加载 YOLOv8 预训练模型 (可以使用 'yolov8n.pt', 'yolov8s.pt' 等)
    model = YOLO('/Applications/jetbrains/PyCharm/jiedan/yolov8-mdi-system/yolov8n.pt')

    # 打开摄像头（MacBook Air M2 可能是 1，但如果不行，尝试 0）
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        # 进行 YOLOv8 推理
        results = model(frame)

        # 解析推理结果，并绘制到画面上
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取目标框坐标
                conf = box.conf[0].item()  # 置信度
                cls = int(box.cls[0].item())  # 目标类别索引
                label = f"{model.names[cls]} {conf:.2f}"

                # 画框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 画标签
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示推理后的画面
        cv2.imshow("YOLOv8 Detection", frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭窗口


if __name__ == "__main__":
    # detected_camera()
    start_tome = time.time()
    sleep(1)
    end_time = start_tome - time.time()
    print(f'----->{end_time}')