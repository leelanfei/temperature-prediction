"""
预测模型模块 - models.py
负责建立温度预测模型并进行预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


def load_prepared_data(file_path='../data/weather.csv'):
    """
    加载并准备数据
    
    Parameters:
        file_path: CSV文件路径
        
    Returns:
        DataFrame: 准备好的数据
    """
    from preprocessing import prepare_data
    return prepare_data(file_path)


def build_simple_linear_model(data):
    """
    构建简单线性回归模型（针对最高气温）
    
    Parameters:
        data: 准备好的数据
        
    Returns:
        tuple: (模型, 特征数据, 预测结果)
    """
    X = data[['天数']]
    y = data['最高气温']
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # 评估模型
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print("=" * 50)
    print("简单线性回归模型")
    print("=" * 50)
    print(f"截距: {model.intercept_:.4f}")
    print(f"斜率: {model.coef_[0]:.4f}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"R² 分数: {r2:.4f}")
    print()
    
    return model, X, y_pred


def build_cycle_linear_model(data):
    """
    构建带周期特征的线性回归模型
    
    Parameters:
        data: 准备好的数据
        
    Returns:
        tuple: (模型, 特征数据, 预测结果)
    """
    # 添加周期特征（一年周期）
    data_cycle = data.copy()
    data_cycle['sin_day'] = np.sin(2 * np.pi * data_cycle['天数'] / 365)
    data_cycle['cos_day'] = np.cos(2 * np.pi * data_cycle['天数'] / 365)
    
    X = data_cycle[['天数', 'sin_day', 'cos_day']]
    y = data_cycle['最高气温']
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # 评估模型
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print("=" * 50)
    print("周期特征线性回归模型")
    print("=" * 50)
    print(f"截距: {model.intercept_:.4f}")
    print(f"系数 (天数): {model.coef_[0]:.6f}")
    print(f"系数 (sin): {model.coef_[1]:.4f}")
    print(f"系数 (cos): {model.coef_[2]:.4f}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"R² 分数: {r2:.4f}")
    print()
    
    return model, X, y_pred


def build_arima_model(data, order=(2, 1, 2)):
    """
    构建ARIMA时间序列模型
    
    Parameters:
        data: 准备好的数据（最高气温序列）
        order: ARIMA order参数 (p, d, q)
        
    Returns:
        tuple: (模型, 结果)
    """
    y = data['最高气温']
    
    model = ARIMA(y, order=order)
    result = model.fit()
    
    print("=" * 50)
    print(f"ARIMA{order} 模型")
    print("=" * 50)
    print(result.summary())
    print()
    
    return model, result


def predict_future_days(model, data, days=7):
    """
    预测未来N天的温度
    
    Parameters:
        model: 周期线性回归模型
        data: 历史数据
        days: 预测天数
        
    Returns:
        array: 预测结果
    """
    # 最后一天的天数
    last_day = data['天数'].max()
    
    # 生成未来天数
    future_days = np.arange(last_day + 1, last_day + 1 + days)
    
    # 计算周期特征
    future_sin = np.sin(2 * np.pi * future_days / 365)
    future_cos = np.cos(2 * np.pi * future_days / 365)
    
    # 构建特征矩阵
    X_future = np.column_stack([future_days, future_sin, future_cos])
    
    # 预测
    forecast = model.predict(X_future)
    
    return forecast


def predict_arima_future(result, data, steps=7):
    """
    使用ARIMA模型预测未来N天
    
    Parameters:
        result: ARIMA模型结果
        data: 历史数据
        steps: 预测天数
        
    Returns:
        array: 预测结果
    """
    forecast = result.forecast(steps=steps)
    return forecast


def plot_predictions(data, y_pred_lr, y_pred_cycle, forecast_lr=None, save_path=None):
    """
    绘制预测结果对比图
    
    Parameters:
        data: 历史数据
        y_pred_lr: 简单线性回归预测
        y_pred_cycle: 周期回归预测
        forecast_lr: 未来预测（可选）
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(14, 6))
    
    # 绘制原始数据
    plt.plot(data['日期'], data['最高气温'], label='实际温度', color='gray', alpha=0.5)
    
    # 绘制简单线性回归预测
    plt.plot(data['日期'], y_pred_lr, label='简单线性回归', color='blue', alpha=0.7, linestyle='--')
    
    # 绘制周期回归预测
    plt.plot(data['日期'], y_pred_cycle, label='周期回归', color='green', alpha=0.7, linestyle='-.')
    
    # 绘制未来预测
    if forecast_lr is not None:
        last_date = data['日期'].iloc[-1]
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(forecast_lr))
        plt.plot(forecast_dates, forecast_lr, label='未来预测', color='red', linestyle=':', marker='o')
    
    plt.xlabel('日期')
    plt.ylabel('温度 (°C)')
    plt.title('温度预测对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测图已保存至: {save_path}")
    
    plt.tight_layout()
    plt.show()


def evaluate_models(data):
    """
    评估所有模型
    
    Parameters:
        data: 准备好的数据
    """
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 简单线性回归
    model_lr, X_lr, y_pred_lr = build_simple_linear_model(data)
    
    # 周期回归
    model_cycle, X_cycle, y_pred_cycle = build_cycle_linear_model(data)
    
    # ARIMA
    model_arima, result_arima = build_arima_model(data)
    
    print("\n模型比较:")
    print("-" * 50)
    print(f"简单线性回归 R²: {r2_score(data['最高气温'], y_pred_lr):.4f}")
    print(f"周期回归 R²: {r2_score(data['最高气温'], y_pred_cycle):.4f}")
    

def run_predictionPipeline(file_path='../data/weather.csv'):
    """
    运行完整的预测流程
    
    Parameters:
        file_path: CSV文件路径
    """
    print("加载数据...")
    data = load_prepared_data(file_path)
    
    print("\n训练模型...")
    
    # 构建模型
    model_lr, X_lr, y_pred_lr = build_simple_linear_model(data)
    model_cycle, X_cycle, y_pred_cycle = build_cycle_linear_model(data)
    model_arima, result_arima = build_arima_model(data)
    
    # 预测未来7天
    print("\n预测未来7天...")
    forecast_lr = predict_future_days(model_cycle, data, days=7)
    forecast_arima = predict_arima_future(result_arima, data, steps=7)
    
    print(f"周期回归预测: {forecast_lr}")
    print(f"ARIMA预测: {forecast_arima.values}")
    
    # 绘制预测结果
    print("\n绘制预测结果...")
    plot_predictions(data, y_pred_lr, y_pred_cycle, forecast_lr, '../figures/prediction_comparison.png')


if __name__ == '__main__':
    # 运行预测流程
    run_predictionPipeline()