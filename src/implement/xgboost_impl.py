import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def to_dataset(df_daily:pd.DataFrame) -> Tuple:
    # 确保 '发病日期' 是时间格式
    df_daily['发病日期'] = pd.to_datetime(df_daily['发病日期'])

    # 设置索引
    df_daily = df_daily.set_index('发病日期').sort_index()

    # 确保数值列没有缺失值
    df_daily = df_daily.fillna(0)

    df_daily['Target'] = df_daily['今日确诊数'].shift(-1)
    features = [col for col in df_daily.columns if col not in ['Target']]
    # 去掉最后一天（因为它的 Target 为空）
    df_train = df_daily.iloc[:-1].copy()

    # 划分训练集 & 测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df_train[features], df_train['Target'], test_size=0.2, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def train(city:str, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, **kwargs):
    # 定义模型
    model = xgb.XGBRegressor(
        **kwargs
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 预测训练集和测试集
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算误差 (Train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)  # RMSE = sqrt(MSE)
    non_zero_idx_train = y_train != 0  # Select indices where y_train is NOT zero
    mape_train = np.mean(np.abs((y_train[non_zero_idx_train] - y_train_pred[non_zero_idx_train]) / y_train[non_zero_idx_train])) * 100

    # 计算误差 (Test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)  # RMSE = sqrt(MSE)
    non_zero_idx_test = y_test != 0  # Select indices where y_test is NOT zero
    mape_test = np.mean(np.abs((y_test[non_zero_idx_test] - y_test_pred[non_zero_idx_test]) / y_test[non_zero_idx_test])) * 100

    print(city,"result")
    # 打印结果
    print(f"Train Mean Absolute Error (MAE): {mae_train:.4f}")
    print(f"Train Mean Squared Error (MSE): {mse_train:.4f}")
    print(f"Train Root Mean Squared Error (RMSE): {rmse_train:.4f}")
    print(f"Train Mean Absolute Percentage Error (MAPE): {mape_train:.2f}%")

    print(f"Test Mean Absolute Error (MAE): {mae_test:.4f}")
    print(f"Test Mean Squared Error (MSE): {mse_test:.4f}")
    print(f"Test Root Mean Squared Error (RMSE): {rmse_test:.4f}")
    print(f"Test Mean Absolute Percentage Error (MAPE): {mape_test:.2f}%")
    return model


def plot(city:str, model, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 真实值
    y_train_actual = np.array(y_train)
    y_test_actual = np.array(y_test)


    # 真实值 & 预测值
    y_train_actual = np.array(y_train)  # 训练集真实值
    y_train_pred = np.array(y_train_pred)  # 训练集预测值
    y_test_actual = np.array(y_test)  # 测试集真实值
    y_test_pred = np.array(y_test_pred)  # 测试集预测值


    sampling_rate = 30

    plt.figure(figsize=(12, 6))


    plt.plot(range(0, len(y_train_actual), sampling_rate), y_train_actual[::sampling_rate], label="Train Actual", color="blue", alpha=0.7, linestyle="-")
    plt.plot(range(0, len(y_train_pred), sampling_rate), y_train_pred[::sampling_rate], label="Train Predicted", color="green", alpha=0.7, linestyle="--")

    # 测试集
    test_start_idx = len(y_train_actual)
    plt.plot(range(test_start_idx, test_start_idx + len(y_test_actual), sampling_rate), y_test_actual[::sampling_rate], label="Test Actual", color="red", alpha=0.7, linestyle="-")
    plt.plot(range(test_start_idx, test_start_idx + len(y_test_pred), sampling_rate), y_test_pred[::sampling_rate], label="Test Predicted", color="green", alpha=0.7, linestyle="--")
    plt.legend()
    plt.title(f"{city}Actual vs Predicted (Sampled)")
    plt.xlabel("Time")
    plt.ylabel("Cases")
    plt.show()

def get_importance(model):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题
    # 获取特征重要性
    xgb.plot_importance(model, max_num_features=25)  # 只显示前20个最重要的
    plt.show()

def get_related(city,X_train):
    # 计算特征的相关性矩阵
    corr_matrix = X_train.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
    print(city)
    plt.show()