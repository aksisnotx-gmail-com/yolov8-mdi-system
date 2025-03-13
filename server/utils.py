import base64
import datetime
import json
import logging
import os
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from cfg import UPLOADS_FOLDER_PATH, RESULTS_FOLDER_PATH, LOGS_FOLDER_PATH, DATASETS_FOLDER_PATH, \
    MODELS_FOLDER_PATH


# # 设置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(SYS_LOG_PATH),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger("MDI-System")

logger = logging.getLogger("MDI-System")

def setup_directories():
    """创建必要的目录结构"""
    # dirs = ["uploads", "results", "logs", "datasets", "models"]
    dirs = [UPLOADS_FOLDER_PATH, RESULTS_FOLDER_PATH, LOGS_FOLDER_PATH, DATASETS_FOLDER_PATH, MODELS_FOLDER_PATH]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f'创建{dir_name}目录成功')
    logger.info("<----初始化目录---->")

def get_file_info(file) -> Dict:
    """获取上传文件的信息"""
    if file is None:
        return {}
    
    file_details = {
        "文件名": file.name,
        "文件类型": file.type,
        "文件大小(MB)": round(file.size / (1024 * 1024), 2)
    }
    return file_details

def save_uploaded_file(uploaded_file, save_dir=UPLOADS_FOLDER_PATH) -> str:
    """保存上传的文件并返回保存路径"""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    logger.info(f"文件已保存至: {file_path}")
    return file_path

def is_video_file(file_path: str) -> bool:
    """判断文件是否为视频"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    _, ext = os.path.splitext(file_path.lower())
    return ext in video_extensions

def is_image_file(file_path: str) -> bool:
    """判断文件是否为图片"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    _, ext = os.path.splitext(file_path.lower())
    return ext in image_extensions

def create_detection_log(file_name: str, detection_results: Dict, inference_time: float) -> Dict:
    """创建检测日志"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "file_name": file_name,
        "inference_time_ms": round(inference_time * 1000, 2),
        "detection_results": detection_results
    }
    
    # 将日志保存到文件
    log_file = os.path.join(LOGS_FOLDER_PATH, "detection_logs.json")
    # os.makedirs("logs", exist_ok=True)
    
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"保存日志时出错: {str(e)}")
    
    return log_entry

def generate_detection_statistics(detection_results: Dict) -> pd.DataFrame:
    """生成检测统计数据"""
    if not detection_results:
        return pd.DataFrame()
    
    data = []
    for class_name, count in detection_results.items():
        data.append({"类别": class_name, "数量": count})
    
    return pd.DataFrame(data)

def export_to_csv(data: pd.DataFrame, filename: str = "detection_report.csv") -> str:
    """导出数据为CSV文件"""
    filepath = os.path.join(RESULTS_FOLDER_PATH, filename)
    # os.makedirs("results", exist_ok=True)
    data.to_csv(filepath, index=False, encoding='utf-8-sig')
    return filepath

def export_to_json(data: Dict, filename: str = "detection_report.json") -> str:
    """导出数据为JSON文件"""
    filepath = os.path.join(RESULTS_FOLDER_PATH, filename)
    # os.makedirs("results", exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return filepath

def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    """创建条形图"""
    fig = px.bar(df, x=x_col, y=y_col, title=title)
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white"
    )
    return fig

def create_pie_chart(df: pd.DataFrame, names_col: str, values_col: str, title: str) -> go.Figure:
    """创建饼图（修正版）"""
    # 校验输入
    if df.empty:
        return go.Figure()
    if names_col not in df.columns or values_col not in df.columns:
        raise ValueError(f"列名错误: {names_col} 或 {values_col} 不存在")

    df = df.copy()

    # 强制转换数值类型并清理数据
    df[values_col] = pd.to_numeric(df[values_col], errors="coerce")
    df = df.dropna(subset=[values_col])

    total = df[values_col].sum()
    if total == 0:
        return go.Figure()

    # 生成饼图
    go_pie = go.Pie(labels=list(df[names_col]), values=list(df[values_col]),
                    hovertemplate="<b>%{label}</b><br>数量: %{value}<br>百分比: %{percent}%<extra></extra>",
                    texttemplate="%{label}<br>%{percent}%",
                    textposition="inside")
    fig = go.Figure(data=[go_pie])

    fig.update_layout(
        title=title,
        template="plotly_white",
        legend_title="类别",
        uniformtext_minsize=12,
        uniformtext_mode="hide"
    )
    return fig


def get_file_download_link(file_path: str, link_text: str) -> str:
    """生成文件下载链接"""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    extension = os.path.splitext(file_path)[1]
    
    if extension == '.csv':
        mime_type = 'text/csv'
    elif extension == '.json':
        mime_type = 'application/json'
    else:
        mime_type = 'application/octet-stream'
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href 