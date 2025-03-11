#!/bin/bash

# 设定容器名称
CONTAINER_NAME="yolov8-mdi-system"
IMAGE_NAME="python:3.9"  # 你可以更改 Python 版本
HOST_PORT=8501  # Streamlit 默认端口
CONTAINER_PORT=8501
PROJECT_DIR="/home/servers/yolov8-mdi-system-server/yolov8-mdi-system"  # 你的项目路径

# 1. 停止并删除已有容器（如果存在）
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo ">>> 停止并删除已有容器..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# 2. 运行 Docker 容器
echo ">>> 启动 Docker 容器..."
docker run -d --name $CONTAINER_NAME \
    -p $HOST_PORT:$CONTAINER_PORT \
    -v $PROJECT_DIR:/app \
    -w /app \
    $IMAGE_NAME \
    tail -f /dev/null
echo ">>> 部署完成！访问 http://localhost:$HOST_PORT 进行测试"
