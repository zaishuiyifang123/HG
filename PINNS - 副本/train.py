import matplotlib
matplotlib.use('Agg')  # 新增的后端设置
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from matplotlib import pyplot as plt
import model
import time
import shift

# 参数设置
dim = 128
Steps = 50000  # 迭代步数
LR = 0.0001  # 学习率
noise_level = 1.0 / 30

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建结果保存路径
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

result_save_path = './results/'
model_save_path = './models/'
weight_path = 'model.pth'
mkdir(result_save_path)
mkdir(model_save_path)
# 文件夹路径定义
input_save_path = os.path.join(result_save_path, 'input_images')
phase_save_path = os.path.join(result_save_path, 'phase_output')
recon_save_path = os.path.join(result_save_path, 'reconstructed_input')

# 创建三个子文件夹
mkdir(input_save_path)
mkdir(phase_save_path)
mkdir(recon_save_path)
# 使用固定路径保存最佳模型
best_model_path = os.path.join(model_save_path, 'best_model.pth')

# 加载原始数据
# HG_path = os.path.join("", f"HmGS10_.npz")
# measure_temp = np.load(HG_path)['arr_0']
# measure_temp = Image.open("HG10.png").convert('L')  # 'L' 表示灰度模式
# measure_temp = np.abs(np.array(measure_temp))  # 转为NumPy数组
# measure_temp = measure_temp / np.max(measure_temp)
# # 修改后的图像加载部分
measure_temp = Image.open("HG10.png").convert('L').resize((128, 128))  # 增加resize
measure_temp = np.abs(np.array(measure_temp))
measure_temp = measure_temp / np.max(measure_temp)
print(measure_temp.shape)  # 应该输出(128, 128)
plt.imshow(measure_temp, cmap='gray')
plt.savefig("HG.png")
measure_temp = torch.tensor(measure_temp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# 定义模型
model = model.UNet().to(device)
if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path))
    print('成功加载预训练权重!')
else:
    print("未找到预训练权重，从头开始训练!")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 定义学习率调整器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10000, verbose=True)

# 训练过程
print('开始训练...')
m_loss = np.zeros(Steps)
start_time = time.time()
min_loss = float('inf')  # 初始化最小loss
best_step = -1  # 最佳模型对应的步数

for step in range(Steps):
    # 准备输入数据
    new_rand = torch.rand((1, 1, dim, dim), device=device, dtype=torch.float32) * noise_level
    input_data = measure_temp + new_rand

    # 前向传播
    optimizer.zero_grad()
    output = model(input_data)
    out_measure = shift.phase_to_amp_ones(output)
    loss = criterion(out_measure, input_data)

    # 反向传播
    loss.backward()
    optimizer.step()

    # 记录损失
    m_loss[step] = loss.item()

    # 检查并更新最佳模型
    if loss.item() < min_loss:
        min_loss = loss.item()
        best_step = step
        torch.save(model.state_dict(), best_model_path)
        print(f'发现新的最佳模型，步数：{step}，loss：{min_loss:.6f}')

    # 定期打印训练信息
    if step % 100 == 0:
        current_loss = loss.item()

        # 保存输入图像
        plt.figure()
        plt.imshow(input_data.squeeze().detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(
            os.path.join(input_save_path, f'input_step{step}.png'),
            bbox_inches='tight',
            pad_inches=0,
            dpi=300
        )
        plt.close()

        # 保存相位输出图像
        plt.figure()
        plt.imshow(output.squeeze().detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(
            os.path.join(phase_save_path, f'phase_step{step}_loss{current_loss:.6f}.png'),
            bbox_inches='tight',
            pad_inches=0,
            dpi=300
        )
        plt.close()

        # 保存重建图像
        plt.figure()
        recon_img = shift.phase_to_amp_ones(output).squeeze().detach().cpu().numpy()
        plt.imshow(recon_img, cmap='gray')
        plt.axis('off')
        plt.savefig(
            os.path.join(recon_save_path, f'recon_step{step}_loss{current_loss:.6f}.png'),
            bbox_inches='tight',
            pad_inches=0,
            dpi=300
        )
        plt.close()

    # 更新学习率
    scheduler.step(loss.item())

# 训练结束处理
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

# 保存loss数据
loss_path = os.path.join(result_save_path, 'loss.npy')
np.save(loss_path, m_loss)

# 绘制loss曲线（已去除网格线）
plt.figure(figsize=(10, 6))
plt.plot(m_loss)
plt.scatter(best_step, min_loss, color='red', label=f'Best: {min_loss:.4f} @ step {best_step}')
plt.xlabel('Training Steps')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig(os.path.join(result_save_path, 'loss_curve.png'))  # 移除了grid(True)
plt.close()

# 保存训练日志
log_path = os.path.join(result_save_path, 'training_log.txt')
with open(log_path, 'w') as f:
    f.write(f'====== 训练参数 ======\n')
    f.write(f'总步数: {Steps}\n')
    f.write(f'学习率: {LR}\n')
    f.write(f'噪声水平: {noise_level}\n')
    f.write(f'设备: {device}\n\n')

    f.write(f'====== 训练结果 ======\n')
    f.write(f'总训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒\n')
    f.write(f'最终Loss: {m_loss[-1]:.6f}\n')
    f.write(f'最佳Loss: {min_loss:.6f} (第 {best_step} 步)\n')
    f.write(f'学习率最终值: {optimizer.param_groups[0]["lr"]:.2e}\n\n')

    f.write(f'====== 文件路径 ======\n')
    f.write(f'最佳模型路径: {best_model_path}\n')
    f.write(f'Loss数据文件: {loss_path}\n')
    f.write(f'Loss曲线图: {os.path.join(result_save_path, "loss_curve.png")}\n')

print('训练完成!')
print(f'最佳模型已保存: {best_model_path}')
print(f'最佳Loss: {min_loss:.6f} (第 {best_step} 步)')
print(f'总训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒')