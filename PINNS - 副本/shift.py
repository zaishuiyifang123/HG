import numpy as np
import torch
import torch.fft


def phase_to_amp_ones(input_phase):
    gs = torch.ones_like(input_phase)  # 假设gs是一个全1的张量
    image = gs * torch.exp(1j * input_phase)
    image = torch.fft.ifft2(torch.fft.fftshift(image))
    image = torch.abs(image)
    image = image / image.max()
    return image


# def gerchberg_saxton(object_amp, target_amp, num_iterations):
#     """
#     Gerchberg-Saxton算法的实现，使用张量计算并返回张量。
#
#     参数:
#     - object_amp: 目标物体的幅度张量。
#     - target_amp: 目标傅里叶空间中的幅度张量。
#     - num_iterations: 算法迭代次数。
#
#     返回:
#     - 最终估计的相位张量。
#     """
#     # 初始化随机相位
#     object_phase = (2 * torch.rand_like(object_amp) - 1) * np.pi
#
#     for _ in range(num_iterations):
#         # 计算物空间场
#         object_field = object_amp * torch.exp(1j * object_phase)
#
#         # 傅里叶变换到频域
#         target_field = torch.fft.fftn(object_field, dim=(-2, -1))
#
#         # 更新频域相位信息
#         temp_phase = torch.angle(target_field)
#         target_field = target_amp * torch.exp(1j * temp_phase)
#
#         # 逆傅里叶变换回到物空间
#         object_field = torch.fft.ifftn(target_field, dim=(-2, -1))
#
#         # 更新物空间相位信息
#         object_phase = torch.angle(object_field)
#
#     return object_phase


def gerchberg_saxton(object_amp, target_amp, num_iterations):
    object_phase = (2 * torch.rand_like(object_amp) - 1) * torch.pi
    for _ in range(num_iterations):
        object_field = object_amp * torch.exp(1j * object_phase)
        target_field = torch.fft.fft2(object_field)
        temp_phase = torch.angle(target_field)
        target_field = target_amp * torch.exp(1j * temp_phase)
        object_field = torch.fft.ifft2(target_field)
        object_phase = torch.angle(object_field)
    return object_phase
