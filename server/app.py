import os
import sys

from ui import main

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # 运行主函数
    main() 