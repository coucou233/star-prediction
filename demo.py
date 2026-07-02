import torch
import pandas as pd
from bpr_model import BPRModel
print("所有依赖安装成功！代码语法无误！")
# 甚至可以随机初始化一个模型看看前向传播
model = BPRModel(用户数, 电影数, 64) # 参数随意填
print("模型初始化成功，环境完全可用！")
