import os
import model
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import shift
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F


model_path = 'models/exp_step_10000_lr_0.000100_noise0.03333333333333333_学习优化.pth'

# 定义数据预处理步骤
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = model.UNet().to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print('successful load weight!')
else:
    print("not successful load weight!")
model.eval()  # 设置模型为评估模式

# 读取输入图片
HG_path = os.path.join("D:\\", f"HmGS10_0.6.npz")
HG = np.load(HG_path)['arr_0']
HG = HG/HG.max()
new_rand = torch.rand((1, 1, 128, 128), device=device, dtype=torch.float32) * 1/30
input_HG = torch.tensor(HG, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
input_batch = input_HG + new_rand

# 将输入传递给模型
with torch.no_grad():  # 不计算梯度
    output = model(input_batch)
    recover = shift.phase_to_amp_ones(output)
# 处理输出
output_image = output.squeeze().detach().cpu().numpy()
recover_image = recover.squeeze().detach().cpu().numpy()

# 计算损失值
mse_loss = F.mse_loss(input_HG, recover).item()
print(f'MSE Loss: {mse_loss}')
mse_loss_np = np.mean((HG-recover_image)**2)
print(f'np MSE Looss:{mse_loss_np}')


# 显示结果
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(HG, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Model Output')
plt.imshow(output_image, cmap='gray')  # 如果输出是单通道图像
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Recovered Image')
plt.imshow(recover_image, cmap='gray')  # 如果输出是单通道图像
plt.axis('off')

plt.show()