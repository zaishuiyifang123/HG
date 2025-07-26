import numpy as np
import pandas as pd

# 加载 .npy 文件
data = np.load('loss.npy', allow_pickle=True)

# 将 numpy 数组转换为 DataFrame
df = pd.DataFrame(data)

# 保存为 Excel 文件
df.to_excel('loss_per_epoch.xlsx', index=False, header=False)