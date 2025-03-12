import logging
import os.path

SYS_LOG_PATH = "server/sys-logs/sys.log"
LOGS_FOLDER_PATH = 'server/logs'
RESULTS_FOLDER_PATH = 'server/results'
UPLOADS_FOLDER_PATH = 'server/uploads'
DATASETS_FOLDER_PATH = 'server/datasets'
MODELS_FOLDER_PATH = 'server/models'
DEFAULT_MODEL_FILE_PATH = '/Applications/jetbrains/PyCharm/jiedan/yolov8-mdi-system/yolov8n.pt'


#初始化日志
if not os.path.exists("server/sys-logs"):
    os.makedirs("server/sys-logs")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(SYS_LOG_PATH),logging.StreamHandler()]
)
logger = logging.getLogger("MDI-System")
logger.info("启动海洋垃圾检测系统")