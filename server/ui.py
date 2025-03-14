import json
import os
from datetime import datetime, timedelta

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils
# 导入自定义模块
from inference import MarineDebrisDetector

# 设置页面配置
st.set_page_config(
    page_title="海洋垃圾检测系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        background-color: #F0F7FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1E88E5;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #4CAF50;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #FFC107;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #F44336;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0D47A1;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
def init_session_state():
    """初始化会话状态变量"""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = None
    if 'inference_time' not in st.session_state:
        st.session_state.inference_time = 0
    if 'class_counts' not in st.session_state:
        st.session_state.class_counts = {}
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "上传检测"
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'dataset_metrics' not in st.session_state:
        st.session_state.dataset_metrics = None

def load_model():
    """加载YOLOv8模型"""
    with st.spinner("正在加载模型，请稍候..."):
        try:
            st.session_state.detector = MarineDebrisDetector()
            st.session_state.model_loaded = True
            st.success("模型加载成功！")
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            st.session_state.model_loaded = False

def display_header():
    """显示页面标题和介绍"""
    st.markdown('<h1 class="main-header">🌊 海洋垃圾检测系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">基于YOLOv8的海洋垃圾检测与分析系统，支持图像、视频和实时摄像头检测。</p>', unsafe_allow_html=True)

def display_sidebar():
    """显示侧边栏"""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png", use_column_width=True)
        st.markdown("## ")
        
        # 导航选项
        tabs = {
            "上传检测": "📤 上传图片/视频检测",
            "实时检测": "📹 实时摄像头检测",
            "检测日志": "📊 检测结果与统计",
            "数据集管理": "🗃️ 数据集管理与训练"
        }
        
        for tab_id, tab_name in tabs.items():
            if st.button(tab_name, key=f"btn_{tab_id}"):
                st.session_state.current_tab = tab_id
        
        st.markdown("---")
        st.markdown("## 关于")
        st.markdown("本系统使用YOLOv8进行海洋垃圾检测，支持多种垃圾类型的识别与统计。")
        st.markdown("版本: v1.0.0")
        st.markdown("---")
        # st.markdown("Made with ❤️ by AI Team")

def upload_detection_tab():
    """上传检测标签页"""
    st.markdown('<h2 class="sub-header">📤 上传图片/视频进行检测</h2>', unsafe_allow_html=True)
    
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 文件上传
        st.markdown('<p class="info-text">请上传图片或视频文件:</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("选择文件", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # 显示文件信息
            file_details = utils.get_file_info(uploaded_file)
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("### 文件信息")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 检测参数设置
            st.markdown("### 检测参数")
            conf_threshold = st.slider("置信度阈值", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
            iou_threshold = st.slider("NMS IOU阈值", min_value=0.1, max_value=1.0, value=0.45, step=0.05)
            
            # 检测按钮
            if st.button("开始检测", key="detect_btn"):
                if not st.session_state.model_loaded:
                    load_model()
                
                if st.session_state.model_loaded:
                    with st.spinner("正在处理，请稍候..."):
                        try:
                            # 保存上传的文件
                            file_path = utils.save_uploaded_file(uploaded_file)
                            
                            # 根据文件类型进行检测
                            if utils.is_image_file(file_path):
                                # 图像检测
                                processed_img, class_counts, inference_time = st.session_state.detector.detect_image(
                                    file_path, conf_threshold, iou_threshold
                                )
                                
                                # 保存结果到会话状态
                                st.session_state.processed_image = processed_img
                                st.session_state.class_counts = class_counts
                                st.session_state.inference_time = inference_time
                                st.session_state.detection_results = {
                                    "type": "image",
                                    "file_path": file_path,
                                    "class_counts": class_counts,
                                    "inference_time": inference_time
                                }
                                
                                # 创建检测日志
                                log_entry = utils.create_detection_log(
                                    os.path.basename(file_path), class_counts, inference_time
                                )
                                st.session_state.log_entries.append(log_entry)
                                
                                st.success(f"检测完成！共检测到 {sum(class_counts.values())} 个目标，耗时 {inference_time:.3f} 秒")
                            
                            elif utils.is_video_file(file_path):
                                # 视频检测
                                output_path, class_counts, avg_inference_time,total_inference_time = st.session_state.detector.detect_video(
                                    file_path, None, conf_threshold, iou_threshold
                                )
                                
                                # 保存结果到会话状态
                                st.session_state.processed_video = output_path
                                st.session_state.class_counts = class_counts
                                st.session_state.inference_time = total_inference_time
                                st.session_state.detection_results = {
                                    "type": "video",
                                    "file_path": file_path,
                                    "output_path": output_path,
                                    "class_counts": class_counts,
                                    "inference_time": total_inference_time
                                }
                                
                                # 创建检测日志
                                log_entry = utils.create_detection_log(
                                    os.path.basename(file_path), class_counts, total_inference_time
                                )
                                st.session_state.log_entries.append(log_entry)
                                
                                st.success(f"视频处理完成！共检测到 {sum(class_counts.values())} 个目标，平均推理时间 {avg_inference_time:.3f} 秒/帧")
                            
                            else:
                                st.error("不支持的文件类型！")
                        
                        except Exception as e:
                            st.error(f"检测过程中出错: {str(e)}")
    
    with col2:
        # 显示检测结果
        if st.session_state.detection_results is not None:
            st.markdown('<h3 class="sub-header">检测结果</h3>', unsafe_allow_html=True)
            
            # 根据检测类型显示结果
            if st.session_state.detection_results["type"] == "image" and st.session_state.processed_image is not None:
                # 显示处理后的图像
                st.image(
                    cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB),
                    caption="检测结果图像",
                    use_column_width=True
                )

            elif st.session_state.detection_results["type"] == "video" and st.session_state.processed_video is not None:
                # 显示处理后的视频 streamlit无法展示OpenCV 重新组装的视频原因在于视频格式解码器
                # video_file = open(st.session_state.processed_video, 'rb')
                # video_bytes = video_file.read()
                st.video(st.session_state.processed_video)
            # 显示检测统计
            if st.session_state.class_counts:
                st.markdown('<h3 class="sub-header">检测统计</h3>', unsafe_allow_html=True)
                
                # 创建统计数据
                df = utils.generate_detection_statistics(st.session_state.class_counts)
                
                # 显示统计表格
                st.dataframe(df, use_container_width=True)

                # 创建条形图
                fig = utils.create_bar_chart(df, "类别", "数量", "垃圾类型分布")
                st.plotly_chart(fig, use_container_width=True)

                # 创建饼图
                fig2 = utils.create_pie_chart(df, "类别", "数量", "垃圾类型占比")
                st.plotly_chart(fig2, use_container_width=True)
                
                # 导出选项
                st.markdown("### 导出结果")
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    if st.button("导出为CSV"):
                        csv_path = utils.export_to_csv(df)
                        st.markdown(utils.get_file_download_link(csv_path, "下载CSV文件"), unsafe_allow_html=True)
                
                with col_exp2:
                    if st.button("导出为JSON"):
                        json_path = utils.export_to_json(st.session_state.detection_results)
                        st.markdown(utils.get_file_download_link(json_path, "下载JSON文件"), unsafe_allow_html=True)

def realtime_detection_tab():
    """实时检测标签页"""
    st.markdown('<h2 class="sub-header">📹 实时摄像头检测</h2>', unsafe_allow_html=True)
    
    # 检查模型是否已加载
    if not st.session_state.model_loaded:
        st.warning("请先加载模型")
        if st.button("加载模型"):
            load_model()
        return
    
    # 摄像头设置
    st.markdown("### 摄像头设置")
    camera_id = st.number_input("摄像头ID", min_value=0, max_value=10, value=0, step=1)
    
    # 检测参数设置
    st.markdown("### 检测参数")
    conf_threshold = st.slider("置信度阈值", min_value=0.1, max_value=1.0, value=0.25, step=0.05, key="rt_conf")
    iou_threshold = st.slider("NMS IOU阈值", min_value=0.1, max_value=1.0, value=0.45, step=0.05, key="rt_iou")
    
    # 实时检测按钮
    st.markdown('<div class="warning-box">点击"开始实时检测"后，将在下方显示实时检测画面。点击"停止检测"按钮可停止检测。</div>', unsafe_allow_html=True)
    
    if st.button("开始实时检测", key="start_webcam"):
        try:
            # 调用实时检测函数
            class_counts = st.session_state.detector.detect_webcam(camera_id, conf_threshold, iou_threshold)
            
            # 保存结果到会话状态
            st.session_state.class_counts = class_counts
            
            # 创建检测日志
            log_entry = utils.create_detection_log(
                f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                class_counts, 
                0
            )
            st.session_state.log_entries.append(log_entry)
            
            st.success(f"实时检测已完成！共检测到 {sum(class_counts.values())} 个目标")
            
            # 显示检测统计
            if class_counts:
                st.markdown('<h3 class="sub-header">检测统计</h3>', unsafe_allow_html=True)
                
                # 创建统计数据
                df = utils.generate_detection_statistics(class_counts)
                
                # 显示统计表格
                st.dataframe(df, use_container_width=True)
                
                # 创建条形图
                fig = utils.create_bar_chart(df, "类别", "数量", "垃圾类型分布")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"实时检测过程中出错: {str(e)}")

def detection_logs_tab():
    """检测日志标签页"""
    st.markdown('<h2 class="sub-header">📊 检测结果与统计</h2>', unsafe_allow_html=True)
    
    # 尝试加载日志文件
    log_file = os.path.join("logs", "detection_logs.json")

    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            # 更新会话状态中的日志
            st.session_state.log_entries = logs
        except Exception as e:
            st.error(f"加载日志文件时出错: {str(e)}")
    
    # 显示日志
    if st.session_state.log_entries:
        # 创建日志数据框
        log_data = []
        for log in st.session_state.log_entries:
            total_objects = sum(log["detection_results"].values()) if log["detection_results"] else 0
            log_data.append({
                "时间戳": log["timestamp"],
                "文件名": log["file_name"],
                "推理时间(s)": log["inference_time_ms"],
                "检测目标数": total_objects
            })
        
        log_df = pd.DataFrame(log_data).sort_values(by="时间戳", ascending=False)

        # 显示日志表格
        st.markdown("### 检测日志")
        st.dataframe(log_df, use_container_width=True)
        
        # 创建时间序列图
        st.markdown("### 检测趋势")

        # 创建并显示图表
        fig = go.Figure(go.Scatter(x=[i["时间戳"] for i in log_data],
                                   y=[i["检测目标数"] for i in log_data],
                                   mode='lines+markers',
                                   hovertemplate='时间：%{x|%Y-%m-%d %H:%M:%S}<br>目标数：%{y}'))
        fig.update_layout(title='检测目标数量趋势', xaxis_title='时间戳', yaxis_title='检测目标数')

        # 创建时间序列图
        st.plotly_chart(fig, use_container_width=True)


        # 创建推理时间图
        fig2 = go.Figure(go.Scatter(x=[i["时间戳"] for i in log_data],
                                   y=[i["推理时间(s)"] for i in log_data],
                                   mode='lines+markers',
                                   hovertemplate='时间：%{x|%Y-%m-%d %H:%M:%S}<br>推理时间(s)：%{y}'))
        fig2.update_layout(title='推理时间趋势', xaxis_title='时间戳', yaxis_title='推理时间(s)')
        st.plotly_chart(fig2, use_container_width=True)


        # 统计所有检测的类别分布
        all_class_counts = {}
        for log in st.session_state.log_entries:
            for cls, count in log["detection_results"].items():
                if cls in all_class_counts:
                    all_class_counts[cls] += count
                else:
                    all_class_counts[cls] = count
        
        if all_class_counts:
            st.markdown("### 累计检测统计")
            
            # 创建统计数据
            df = utils.generate_detection_statistics(all_class_counts)
            
            # 显示统计表格
            st.dataframe(df, use_container_width=True)
            
            # 创建条形图
            fig3 = utils.create_bar_chart(df, "类别", "数量", "累计垃圾类型分布")
            st.plotly_chart(fig3, use_container_width=True)
            
            # 创建饼图
            fig4 = utils.create_pie_chart(df, "类别", "数量", "累计垃圾类型占比")
            st.plotly_chart(fig4, use_container_width=True)
            
            # 导出选项
            st.markdown("### 导出统计")
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                if st.button("导出累计统计为CSV"):
                    csv_path = utils.export_to_csv(df, "cumulative_statistics.csv")
                    st.markdown(utils.get_file_download_link(csv_path, "下载CSV文件"), unsafe_allow_html=True)
            
            with col_exp2:
                if st.button("导出所有日志为JSON"):
                    json_path = utils.export_to_json(st.session_state.log_entries, "all_detection_logs.json")
                    st.markdown(utils.get_file_download_link(json_path, "下载JSON文件"), unsafe_allow_html=True)
    else:
        st.info("暂无检测日志")

def dataset_management_tab():
    """数据集管理标签页"""
    st.markdown('<h2 class="sub-header">🗃️ 数据集管理与训练</h2>', unsafe_allow_html=True)
    
    # 检查模型是否已加载
    if not st.session_state.model_loaded:
        st.warning("请先加载模型")
        if st.button("加载模型"):
            load_model()
        return
    
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 上传数据集")
        st.markdown('<div class="info-text">请上传YOLO格式的数据集配置文件(yaml):</div>', unsafe_allow_html=True)
        
        uploaded_dataset = st.file_uploader("选择数据集配置文件", type=["yaml", "yml"])
        
        if uploaded_dataset is not None:
            # 显示文件信息
            file_details = utils.get_file_info(uploaded_dataset)
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("#### 数据集信息")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 保存上传的文件
            dataset_path = utils.save_uploaded_file(uploaded_dataset, "datasets")
            
            st.markdown("### 训练参数")
            epochs = st.slider("训练轮数", min_value=1, max_value=100, value=10, step=1)
            batch_size = st.slider("批次大小", min_value=1, max_value=64, value=16, step=1)
            img_size = st.slider("图像大小", min_value=320, max_value=1280, value=640, step=32)
            
            # 训练按钮
            if st.button("开始训练", key="train_btn"):
                with st.spinner("正在训练模型，这可能需要一些时间..."):
                    try:
                        # 调用训练函数
                        metrics = st.session_state.detector.train_model(
                            dataset_path, epochs, batch_size, img_size
                        )
                        
                        # 保存指标到会话状态
                        st.session_state.dataset_metrics = metrics
                        
                        st.success("模型训练完成！")
                    except Exception as e:
                        st.error(f"模型训练过程中出错: {str(e)}")
        
        st.markdown("### 模型评估")
        st.markdown('<div class="info-text">请上传YOLO格式的数据集配置文件(yaml)进行评估:</div>', unsafe_allow_html=True)
        
        eval_dataset = st.file_uploader("选择评估数据集配置文件", type=["yaml", "yml"], key="eval_dataset")
        
        if eval_dataset is not None:
            # 保存上传的文件
            eval_dataset_path = utils.save_uploaded_file(eval_dataset, "datasets")
            
            # 评估按钮
            if st.button("开始评估", key="eval_btn"):
                with st.spinner("正在评估模型，请稍候..."):
                    try:
                        # 调用评估函数
                        metrics = st.session_state.detector.evaluate_model(eval_dataset_path)
                        
                        # 保存指标到会话状态
                        st.session_state.dataset_metrics = metrics
                        
                        st.success("模型评估完成！")
                    except Exception as e:
                        st.error(f"模型评估过程中出错: {str(e)}")
    
    with col2:
        # 显示模型指标
        if st.session_state.dataset_metrics is not None:
            st.markdown('<h3 class="sub-header">模型评估指标</h3>', unsafe_allow_html=True)
            
            # 创建指标数据框
            metrics_data = []
            for metric, value in st.session_state.dataset_metrics.items():
                metrics_data.append({
                    "指标": metric,
                    "值": value
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # 显示指标表格
            st.dataframe(metrics_df, use_container_width=True)
            
            # 创建雷达图
            categories = metrics_df["指标"].tolist()
            values = metrics_df["值"].tolist()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='模型性能'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="模型性能雷达图",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 创建条形图
            fig2 = px.bar(metrics_df, x="指标", y="值", title="模型性能指标")
            st.plotly_chart(fig2, use_container_width=True)
            
            # 导出选项
            if st.button("导出指标为CSV"):
                csv_path = utils.export_to_csv(metrics_df, "model_metrics.csv")
                st.markdown(utils.get_file_download_link(csv_path, "下载CSV文件"), unsafe_allow_html=True)
        else:
            st.info("请先训练或评估模型以查看指标")

def main():
    """主函数"""
    # 初始化会话状态
    init_session_state()
    
    # 创建必要的目录
    utils.setup_directories()
    
    # 显示页面标题和介绍
    display_header()
    
    # 显示侧边栏
    display_sidebar()
    
    # 根据当前标签页显示内容
    if st.session_state.current_tab == "上传检测":
        upload_detection_tab()
    elif st.session_state.current_tab == "实时检测":
        realtime_detection_tab()
    elif st.session_state.current_tab == "检测日志":
        detection_logs_tab()
    elif st.session_state.current_tab == "数据集管理":
        dataset_management_tab()

if __name__ == "__main__":
    main() 