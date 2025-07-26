import numpy as np
import pandas as pd
import os
from datetime import datetime


def npy_to_excel(input_path, output_path=None):
    """
    将numpy的.npy文件转换为Excel文件
    参数:
        input_path: loss.npy文件路径

        output_path: 输出的Excel文件路径(可选)
    返回:
        生成的Excel文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 加载npy文件
    loss_data = np.load(input_path)

    # 创建DataFrame
    df = pd.DataFrame({
        'Step': np.arange(len(loss_data)),
        'Loss': loss_data
    })

    # 设置默认输出路径
    if output_path is None:
        dir_name = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(dir_name, f"{base_name}_{timestamp}.xlsx")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存为Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Training Loss')

        # 添加统计信息到第二个sheet
        stats_df = pd.DataFrame({
            'Metric': ['Min Loss', 'Max Loss', 'Mean Loss', 'Final Loss'],
            'Value': [np.min(loss_data), np.max(loss_data), np.mean(loss_data), loss_data[-1]],
            'Step': [
                np.argmin(loss_data),
                np.argmax(loss_data),
                'N/A',
                len(loss_data) - 1
            ]
        })
        stats_df.to_excel(writer, index=False, sheet_name='Statistics')

        # 获取Excel writer对象的工作簿和工作表
        workbook = writer.book
        worksheet = writer.sheets['Training Loss']

        # 设置列宽
        for column in ['A', 'B']:
            worksheet.column_dimensions[column].width = 15

        # 添加图表（折线图）
        if len(loss_data) > 1:
            chart = workbook.charts.LineChart()
            chart.title = "Training Loss Curve"
            chart.x_axis.title = "Steps"
            chart.y_axis.title = "Loss"

            data_ref = workbook.defined_names.add_defined_name(
                f'=Training Loss!$B$2:$B${len(loss_data) + 1}',
                name='LossData'
            )
            categories_ref = workbook.defined_names.add_defined_name(
                f'=Training Loss!$A$2:$A${len(loss_data) + 1}',
                name='StepData'
            )

            chart.add_data(data_ref, titles_from_data=True)
            chart.set_categories(categories_ref)

            # 将图表插入到统计sheet中
            worksheet_stats = writer.sheets['Statistics']
            worksheet_stats.add_chart(chart, "E2")

    print(f"成功生成Excel文件: {output_path}")
    return output_path


if __name__ == "__main__":
    # 使用示例
    input_file = "./results/loss.npy"  # 替换为你的实际路径
    try:
        output_file = npy_to_excel(input_file)
        print(f"转换完成，文件已保存到: {output_file}")
    except Exception as e:
        print(f"转换失败: {str(e)}")