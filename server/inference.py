import logging
import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st

from cfg import DEFAULT_MODEL_FILE_PATH, MODELS_FOLDER_PATH, RESULTS_FOLDER_PATH

logger = logging.getLogger("MDI-System")

class MarineDebrisDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL_FILE_PATH):
        """
        初始化海洋垃圾检测器
        
        Args:
            model_path: YOLOv8模型路径，默认使用YOLOv8n
        """
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.load_model()
    
    def load_model(self) -> None:
        """加载YOLOv8模型"""
        try:
            # 确保模型目录存在
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # 如果模型文件不存在，则下载预训练模型
            if not os.path.exists(self.model_path):
                logger.info(f"模型文件 {self.model_path} 不存在，将使用预训练模型")
                self.model = YOLO("yolov8n.pt")
                # 保存模型到指定路径
                self.model.save(self.model_path)
            else:
                self.model = YOLO(self.model_path)
            
            # 获取类别名称
            self.class_names = self.model.names
            logger.info(f"模型加载成功，共有 {len(self.class_names)} 个类别")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def detect_image(self, image_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Tuple[np.ndarray, Dict, float]:
        """
        对图像进行目标检测
        
        Args:
            image_path: 图像文件路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU阈值
            
        Returns:
            处理后的图像，检测结果统计，推理时间
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件 {image_path} 不存在")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像文件 {image_path}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行推理
        results = self.model(image, conf=conf_threshold, iou=iou_threshold)
        
        # 计算推理时间
        inference_time = time.time() - start_time
        
        # 获取检测结果
        result = results[0]
        
        # 统计各类别数量
        class_counts = {}
        
        # 在图像上绘制检测框
        annotated_img = image.copy()
        
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            for box in result.boxes:
                # 获取类别ID和置信度
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                
                # 获取类别名称
                cls_name = self.class_names.get(cls_id, f"类别_{cls_id}")
                
                # 更新类别计数
                if cls_name in class_counts:
                    class_counts[cls_name] += 1
                else:
                    class_counts[cls_name] = 1
                
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # 绘制边界框
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制类别标签和置信度
                label = f"{cls_name}: {conf:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        logger.info(f"图像检测完成，检测到 {sum(class_counts.values())} 个目标，耗时 {inference_time:.3f} 秒")
        
        return annotated_img, class_counts, inference_time
    
    def detect_video(self, video_path: str, output_path: str = None, conf_threshold: float = 0.25, 
                    iou_threshold: float = 0.45, skip_frames: int = 0) -> Tuple[str, Dict, float]:
        """
        对视频进行目标检测
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径，如果为None则自动生成
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU阈值
            skip_frames: 跳帧数，每隔多少帧处理一次
            
        Returns:
            处理后的视频路径，检测结果统计，平均推理时间
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件 {video_path} 不存在")
        
        # 如果未指定输出路径，则自动生成
        if output_path is None:
            filename, ext = os.path.splitext(os.path.basename(video_path))
            output_path = os.path.join(RESULTS_FOLDER_PATH, f"{filename}_detected{ext}")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件 {video_path}")
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 统计各类别数量
        class_counts = {}
        
        # 记录总推理时间和处理帧数
        total_inference_time = 0
        processed_frames = 0
        
        # 处理视频帧
        frame_idx = 0
        
        logger.info(f"开始处理视频，总帧数: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 跳帧处理
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                out.write(frame)
                frame_idx += 1
                continue
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行推理
            results = self.model(frame, conf=conf_threshold, iou=iou_threshold)
            
            # 计算推理时间
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            processed_frames += 1
            
            # 获取检测结果
            result = results[0]
            
            # 在图像上绘制检测框
            annotated_frame = frame.copy()
            
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    # 获取类别ID和置信度
                    cls_id = int(box.cls.item())
                    conf = box.conf.item()
                    
                    # 获取类别名称
                    cls_name = self.class_names.get(cls_id, f"类别_{cls_id}")
                    
                    # 更新类别计数
                    if cls_name in class_counts:
                        class_counts[cls_name] += 1
                    else:
                        class_counts[cls_name] = 1
                    
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # 绘制边界框
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制类别标签和置信度
                    label = f"{cls_name}: {conf:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # 写入处理后的帧
            out.write(annotated_frame)
            
            # 更新帧索引
            frame_idx += 1
            
            # 打印进度
            if frame_idx % 10 == 0:
                progress = (frame_idx / total_frames) * 100
                logger.info(f"视频处理进度: {progress:.1f}%")
        
        # 释放资源
        cap.release()
        out.release()
        
        # 计算平均推理时间
        avg_inference_time = total_inference_time / processed_frames if processed_frames > 0 else 0
        
        logger.info(f"视频处理完成，共处理 {processed_frames} 帧，检测到 {sum(class_counts.values())} 个目标，平均推理时间 {avg_inference_time:.3f} 秒/帧")
        
        return output_path, class_counts, avg_inference_time
    
    def detect_webcam(self, camera_id: int = 0, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Dict:
        """
        使用摄像头进行实时目标检测，使用Streamlit显示画面
        
        Args:
            camera_id: 摄像头ID
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU阈值
            
        Returns:
            检测到的类别计数字典
        """
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头 ID: {camera_id}")
        
        logger.info(f"已打开摄像头 ID: {camera_id}")
        
        # 统计各类别数量
        class_counts = {}
        
        # 创建Streamlit占位符用于显示视频帧
        frame_placeholder = st.empty()
        
        # 创建Streamlit占位符用于显示当前帧的检测结果
        info_placeholder = st.empty()
        
        # 创建停止按钮
        stop_button_placeholder = st.empty()
        stop_button = stop_button_placeholder.button("停止检测")
        
        # 记录FPS相关变量
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # 当前帧的类别计数
        current_frame_counts = {}
        
        try:
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    logger.info("无法读取摄像头画面")
                    break
                
                # 执行推理
                results = self.model(frame, conf=conf_threshold, iou=iou_threshold)
                
                # 获取检测结果
                result = results[0]
                
                # 在图像上绘制检测框
                annotated_frame = frame.copy()
                
                # 清空当前帧的类别计数
                current_frame_counts = {}
                
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    for box in result.boxes:
                        # 获取类别ID和置信度
                        cls_id = int(box.cls.item())
                        conf = box.conf.item()
                        
                        # 获取类别名称
                        cls_name = self.class_names.get(cls_id, f"类别_{cls_id}")
                        
                        # 更新类别计数
                        if cls_name in current_frame_counts:
                            current_frame_counts[cls_name] += 1
                        else:
                            current_frame_counts[cls_name] = 1
                        
                        # 更新总类别计数
                        if cls_name in class_counts:
                            class_counts[cls_name] += 1
                        else:
                            class_counts[cls_name] = 1
                        
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # 绘制边界框
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 绘制类别标签和置信度
                        label = f"{cls_name}: {conf:.2f}"
                        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # 计算并显示FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # 在图像上显示FPS
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 将BGR转换为RGB用于Streamlit显示
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # 使用Streamlit显示帧
                frame_placeholder.image(annotated_frame_rgb, caption="实时检测", use_column_width=True)
                
                # 显示当前帧的检测结果
                if current_frame_counts:
                    info_text = "### 当前检测结果\n"
                    for cls_name, count in current_frame_counts.items():
                        info_text += f"- {cls_name}: {count}\n"
                    info_placeholder.markdown(info_text)
                else:
                    info_placeholder.markdown("### 当前检测结果\n未检测到目标")
                
                # 检查是否点击了停止按钮
                stop_button = stop_button_placeholder.button("停止检测", key=f"stop_{time.time()}")
                
                # 添加短暂延迟，避免过高的CPU使用率
                time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"摄像头检测过程中出错: {str(e)}")
        finally:
            # 释放资源
            cap.release()
            logger.info(f"摄像头检测结束，共检测到 {sum(class_counts.values())} 个目标")
        
        return class_counts
    
    def train_model(self, dataset_path: str, epochs: int = 50, batch_size: int = 16, 
                   img_size: int = 640, output_dir: str = MODELS_FOLDER_PATH) -> Dict:
        """
        使用自定义数据集训练模型
        
        Args:
            dataset_path: 数据集路径（YOLO格式）
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 图像大小
            output_dir: 输出目录
            
        Returns:
            训练结果指标
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集路径 {dataset_path} 不存在")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"开始训练模型，数据集: {dataset_path}，轮数: {epochs}")
        
        try:
            # 创建一个新的YOLO模型实例用于训练
            model = YOLO(DEFAULT_MODEL_FILE_PATH)
            
            # 训练模型
            results = model.train(
                data=dataset_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project=output_dir,
                name="marine_debris_detector"
            )
            
            # 获取训练结果
            metrics = {
                "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
                "val_loss": float(results.results_dict.get("val/box_loss", 0))
            }
            
            # 更新当前模型
            self.model_path = os.path.join(output_dir, "runs", "weights", "best.pt")
            self.load_model()
            
            logger.info(f"模型训练完成，最佳模型保存至: {self.model_path}")
            
            return metrics
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            raise
    
    def evaluate_model(self, dataset_path: str) -> Dict:
        """
        评估模型性能
        
        Args:
            dataset_path: 数据集路径（YOLO格式）
            
        Returns:
            评估指标
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集路径 {dataset_path} 不存在")
        
        logger.info(f"开始评估模型，数据集: {dataset_path}")
        
        try:
            # 评估模型
            results = self.model.val(data=dataset_path)
            
            # 获取评估指标
            metrics = {
                "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0))
            }
            
            logger.info(f"模型评估完成，mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50-95']:.4f}")
            
            return metrics
        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            raise 