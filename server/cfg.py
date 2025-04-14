import logging
import os.path
"""
路径使用相对路径，相对路径相对 streamlit run 执行的路径
"""
SYS_LOG_PATH = "sys-logs/sys.log"
LOGS_FOLDER_PATH = 'logs'
RESULTS_FOLDER_PATH = 'results'
UPLOADS_FOLDER_PATH = 'uploads'
DATASETS_FOLDER_PATH = 'datasets'
MODELS_FOLDER_PATH = 'models'
DEFAULT_MODEL_FILE_PATH = '/Applications/jetbrains/PyCharm/jiedan/yolov8-mdi-system/yolov8n.pt'


#初始化日志
if not os.path.exists("sys-logs"):
    os.makedirs("sys-logs")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(SYS_LOG_PATH),logging.StreamHandler()]
)
logger = logging.getLogger("MDI-System")
logger.info("启动海洋垃圾检测系统")