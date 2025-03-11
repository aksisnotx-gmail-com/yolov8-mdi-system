import streamlit as st
import os
import sys
import logging

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入UI模块
from ui import main

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("mdi_system.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("MDI-System")
    logger.info("启动海洋垃圾检测系统")
    
    # 运行主函数
    main() 