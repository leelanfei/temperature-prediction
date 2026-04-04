"""
数据预处理模块 - preprocessing.py
负责数据读取、清洗、特征工程和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_weather_data(file_path='weather.csv'):
    """
    加载天气数据
    
    Parameters:
        file_path: CSV文件路径
        
    Returns:
        DataFrame: 天气数据
    """
    df = pd.read_csv(file_path)
    print(f"已加载数据，共 {len(df)} 条记录")
    return df


def clean_weather_data(df):
    """
    清洗天气数据
    
    Parameters:
        df: 原始天气数据DataFrame
        
    Returns:
        DataFrame: 清洗后的数据
    """
    df_clean = df.copy()
    
    # 1. 去除温度符号 ℃ 并转换为数值
    df_clean['最高气温'] = df_clean['最高气温'].str.replace('℃', '').astype(float)
    df_clean['最低气温'] = df_clean['最低气温'].str.replace('℃', '').astype(float)
    
    # 2. 提取主天气状态（取 ~ 前的部分）
    df_clean['主要天气'] = df_clean['天气'].str.split('~').str[0]
    
    # 3. 从风向字段提取风力等级
    df_clean['风力等级'] = df_clean['风向'].str.extract(r'(\d+)级').astype(float)
    
    # 4. 转换日期格式
    df_clean['日期'] = pd.to_datetime(df_clean['日期'])
    
    print("数据清洗完成")
    return df_clean


def add_time_features(df):
    """
    添加时间特征
    
    Parameters:
        df: 清洗后的数据DataFrame
        
    Returns:
        DataFrame: 添加时间特征后的数据
    """
    df_feat = df.copy()
    
    # 按日期排序
    df_feat = df_feat.sort_values('日期').reset_index(drop=True)
    
    # 计算相对天数（从第一天开始的天数）
    df_feat['天数'] = (df_feat['日期'] - df_feat['日期'].min()).dt.days
    
    # 提取月份
    df_feat['月份'] = df_feat['日期'].dt.month
    
    # 提取星期几（0=周一，6=周日）
    df_feat['星期'] = df_feat['日期'].dt.dayofweek
    
    print("时间特征添加完成")
    return df_feat


def prepare_data(file_path='weather.csv'):
    """
    完整的数据准备流程
    
    Parameters:
        file_path: CSV文件路径
        
    Returns:
        DataFrame: 准备好的数据
    """
    # 加载数据
    df = load_weather_data(file_path)
    
    # 清洗数据
    df = clean_weather_data(df)
    
    # 添加时间特征
    df = add_time_features(df)
    
    # 选择需要的列
    data = df[['日期', '最高气温', '最低气温', '主要天气', '风力等级', '天数', '月份', '星期']].copy()
    
    return data


def plot_monthly_heatmap(data, save_path=None):
    """
    绘制月度气温热力图
    
    Parameters:
        data: 包含月份和气温数据的数据
        save_path: 保存路径（可选）
    """
    # 计算月度平均气温
    monthly_avg = data.groupby('月份')[['最高气温', '最低气温']].mean()
    
    # 绘制热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(monthly_avg.T, annot=True, fmt=".1f", cmap="YlOrRd", 
               cbar_kws={'label': '温度 (°C)'})
    plt.title('各月平均气温热力图')
    plt.xlabel('月份')
    plt.ylabel('气温类型')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存至: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_trend_with_wind(data, save_path=None):
    """
    绘制气温与风向双轴趋势图
    
    Parameters:
        data: 包含日期、气温和风力等级的数据
        save_path: 保存路径（可选）
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # 左轴：最高气温
    color_red = 'tab:red'
    ax1.set_xlabel('日期')
    ax1.set_ylabel('最高气温 (°C)', color=color_red)
    ax1.plot(data['日期'], data['最高气温'], color=color_red, label='最高气温')
    ax1.tick_params(axis='y', labelcolor=color_red)
    
    # 右轴：风力等级
    ax2 = ax1.twinx()
    color_blue = 'tab:blue'
    ax2.set_ylabel('风力等级', color=color_blue)
    ax2.plot(data['日期'], data['风力等级'], color=color_blue, alpha=0.7, label='风力等级', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_blue)
    
    plt.title('气温与风力等级趋势图')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"趋势图已保存至: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_temperature_trend(data, save_path=None):
    """
    绘制气温趋势图
    
    Parameters:
        data: 包含日期和气温数据的数据
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(14, 6))
    
    plt.plot(data['日期'], data['最高气温'], label='最高气温', color='tab:red', alpha=0.7)
    plt.plot(data['日期'], data['最低气温'], label='最低气温', color='tab:blue', alpha=0.7)
    
    plt.xlabel('日期')
    plt.ylabel('温度 (°C)')
    plt.title('2024年气温趋势图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"趋势图已保存至: {save_path}")
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 示例用法
    print("开始数据预处理...")
    
    # 尝试加载数据（如果文件存在）
    try:
        data = prepare_data('../data/weather.csv')
        print(data.head(10))
        print("\n数据统计:")
        print(data.describe())
        
        # 绘制可视化图表
        plot_temperature_trend(data, '../figures/temperature_trend.png')
        plot_monthly_heatmap(data, '../figures/monthly_heatmap.png')
        plot_trend_with_wind(data, '../figures/trend_with_wind.png')
        
    except FileNotFoundError:
        print("数据文件不存在，请先运行 spider.py 爬取数据")