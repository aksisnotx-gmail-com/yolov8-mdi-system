import base64
import datetime
import json
import logging
import os
import shutil
from typing import Dict

import pandas as pd
import plotly.graph_objects as go

from cfg import UPLOADS_FOLDER_PATH, RESULTS_FOLDER_PATH, LOGS_FOLDER_PATH, DATASETS_FOLDER_PATH, \
    MODELS_FOLDER_PATH

logger = logging.getLogger("MDI-System")

def setup_directories():
    """创建必要的目录结构"""
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
        "inference_time_ms": f'{inference_time:.2f}',
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
    df_sorted = df.sort_values(by=y_col)

    # 创建条形图
    go_bar = go.Bar(
        x=list(df_sorted[x_col]),
        y=list(df_sorted[y_col])
    )

    # 创建 Figure 对象并设置布局
    fig = go.Figure(data=go_bar)
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        title=title,
        template="plotly_white"
    )

    return fig

def create_pie_chart(df: pd.DataFrame, names_col: str, values_col: str, title: str) -> go.Figure:
    """创建饼图"""
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

def clear_folder(dirPath):
    """
    清空目录
    """
    if os.path.exists(dirPath) and os.listdir(dirPath):
        # 文件夹存在且不为空，删除其下所有文件和子目录
        for filename in os.listdir(dirPath):
            file_path = os.path.join(dirPath, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹
            except Exception as e:
                logger.error(f"删除 {file_path} 失败: {e}")
    logger.info(f'{dirPath} 已清空')

def find_first_file_with_suffix(dir_path,suffix: tuple):
    """
    寻找目录下的指定文件类型，也可以是文件名称和后缀，并返回其绝对路径（只返回第一个找到的）
    """
    abspath = get_abspath(dir_path)
    if abspath is None:
        return None

    for root, _, files in os.walk(abspath):
        for file in files:
            if file.endswith(suffix):
                return os.path.join(root, file)
    return None  # 未找到时返回 None

def get_abspath(dir_path):
    """
    返回相对目录/文件的绝对路径。如果路径不存在，返回 None。

    参数:
        dir_path (str): 输入的目录路径（可以是相对路径或绝对路径）

    返回:
        str or None: 绝对路径（如果存在），否则返回 None
    """
    # 将 dir_path 转为绝对路径
    abs_path = os.path.abspath(dir_path)
    # 检查路径是否存在
    if os.path.exists(abs_path):
        return abs_path  # 存在时返回绝对路径
    else:
        return None  # 不存在时返回 None


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


def safe_path(path):
    """对 Windows 添加 \\?\ 前缀，其他平台不处理"""
    if os.name == 'nt':
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(r'\\?\\'):
            return r'\\?\\' + abs_path
        return abs_path
    return path