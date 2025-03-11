from ultralytics import YOLO

"""
直接将分割标注转化为box标注

Roboflow标注网站的开源数据集中收集数据的时候，发现获得的数据集中不会自动分开分割集与检测数据集（Polygon与Bounding Box类型的标注混杂在一起）
导致Yolov8运行的时候自动抛弃分割数据标注部分，导致数据集大量空缺，模型精度大幅下降（Polygon部分被抛弃）
观察txt文件，发现分割的数据集是以坐标形式存在的，而box标注的数据集是以xywh形式存在的。
"""

def conversion_box(org_label_path:str):
    import os

    label_dir = org_label_path

    # 获取路径内的所有文件路径列表
    yolo_file = os.listdir(label_dir)

    # 遍历文件
    for label_name in yolo_file:
        # 打开文件
        label_path = label_dir + "/" + label_name
        file_data = ""
        #判断 path 是否存在
        if not os.path.exists(label_path):
            continue
        with open(label_path, "r") as f:
            for line in f:
                nums = line.split(' ')
                if len(nums) > 5:
                    Head = nums[0]
                    min_X, min_Y, max_X, max_Y = 10, 10, 0, 0
                    for i in range(1, len(nums), 2):
                        if float(nums[i]) < float(min_X):
                            min_X = float(nums[i])
                        if float(nums[i]) > float(max_X):
                            max_X = float(nums[i])
                        if float(nums[i + 1]) < float(min_Y):
                            min_Y = float(nums[i + 1])
                        if float(nums[i + 1]) > float(max_Y):
                            max_Y = float(nums[i + 1])
                    x, y, w, h = (min_X + max_X) / 2, (min_Y + max_Y) / 2, max_X - min_X, max_Y - min_Y
                    line = Head + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + '\n'
                file_data += line

        with open(label_path, "w") as f:
            f.write(file_data)
        print(label_path)


"""
模型训练
"""

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    # 训练（epochs=50 代表训练 50 轮）
    model.train(data="/Applications/jetbrains/PyCharm/jiedan/yolov8-mdi-system/datasets/global.solution.v4i.yolov8/data.yaml", epochs=50,cache=False, imgsz=640,workers=0, batch=4, device="mps",fraction=0.5)
