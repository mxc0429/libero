"""
在训练脚本开头添加这些代码来抑制警告
"""

import warnings
import os

# 抑制所有警告
warnings.filterwarnings('ignore')

# 抑制特定的警告
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 设置环境变量来抑制 robosuite 警告
os.environ['ROBOSUITE_SUPPRESS_WARNINGS'] = '1'
