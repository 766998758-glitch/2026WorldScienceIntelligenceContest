"""
Sklearn GradientBoosting 基线：根据边界条件预测节点电价 A
（纯sklearn实现，无需LightGBM/XGBoost）
气象特征仅保留 ghi_mean 和 ghi_night，已删除 minute 噪声特征
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 路径配置 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')

output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'sklearn_baseline_output.csv')
output_power_path = os.path.join(output_dir, 'output.csv')

feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
target_col = 'A'


# 添加时间特征（已删除 minute）
def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    # df['minute'] = df['times'].dt.minute  # 已删除噪声特征
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df


# 气象数据加载与处理（仅保留 ghi_mean 和 ghi_night）
def load_and_process_nc(filepath):
    import xarray as xr

    ds = xr.open_dataset(filepath)

    u100 = ds['data'].sel(channel='u100')
    v100 = ds['data'].sel(channel='v100')
    ghi  = ds['data'].sel(channel='ghi')

    ws = np.sqrt(u100 ** 2 + v100 ** 2)

    ws_spatial = ws.mean(dim=['lat', 'lon'])
    ghi_spatial = ghi.mean(dim=['lat', 'lon'])

    ws_hourly = ws_spatial.isel(time=0).values
    ghi_hourly = ghi_spatial.isel(time=0).values

    # 只计算需要的统计量
    ghi_mean = np.mean(ghi_hourly)
    ghi_night = np.mean(ghi_hourly[0:6])  # 北京时间 8-13 点

    pub_date = pd.to_datetime(str(ds['time'].values[0]))
    target_date = pd.Timestamp(pub_date + pd.Timedelta(days=1)).date()

    ds.close()

    return pd.DataFrame({
        'date': [target_date],
        'ghi_mean': ghi_mean,
        'ghi_night': ghi_night
    })


def generate_strategy(price_csv, save_path, min_profit_threshold=0):
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])
    df['date'] = df['times'].dt.date
    
    results = []
    total_profit = 0
    
    for date, group in df.groupby('date'):
        prices = group['A'].values
        times = group['times'].values
        
        if len(prices) != 96:
            continue
        
        best_profit = 0
        best_tc = -1
        best_td = -1

        for tc in range(0, 81):
            charge_cost = prices[tc:tc+8].sum() * 1000
            for td in range(tc + 8, 89):
                discharge_revenue = prices[td:td+8].sum() * 1000
                profit = discharge_revenue - charge_cost
                if profit > best_profit:
                    best_profit = profit
                    best_tc = tc
                    best_td = td
        
        power = np.zeros(96)
        # 阈值判断
        if best_profit >= min_profit_threshold and best_profit > 0:
            power[best_tc:best_tc+8] = -1000
            power[best_td:best_td+8] = 1000
            total_profit += best_profit
            print(f"日期: {date}, 充电开始: {best_tc:2d}, 放电开始: {best_td:2d}, 预期收益: {best_profit:10.2f}")
        elif best_profit > 0:
            print(f"日期: {date}, 预测收益 {best_profit:.2f} 低于阈值 {min_profit_threshold}, 不操作")
        else:
            print(f"日期: {date}, 无交易（收益非正）")
        
        for i, (t, p, pr) in enumerate(zip(times, power, prices)):
            results.append({'times': t, '实时价格': pr, 'power': p})
    
    df_result = pd.DataFrame(results)
    df_result.to_csv(save_path, index=False)
    
    n_days = df['date'].nunique()
    avg_profit = total_profit / n_days if n_days > 0 else 0
    print(f'\n充放电策略已保存: {save_path}')
    print(f'总天数: {n_days}')
    print(f'总收益: {total_profit:.2f}')
    print(f'平均日收益: {avg_profit:.2f}')
    
    return df_result


# ==================== 主程序 ====================
if __name__ == '__main__':
    for path in [train_feature_path, train_label_path, test_feature_path]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在: {path}")
            exit(1)

    print("=" * 60)
    print("气象特征精简版：仅 ghi_mean + ghi_night，无 minute 噪声，2000阈值")
    print("=" * 60)

    # 1. 数据准备
    print("\n[1/4] 加载数据...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)
    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])

    # 加载气象数据
    import xarray as xr
    from glob import glob

    nc_dir = '/mnt/workspace/all_nc/to_sais_new/all_nc'
    nc_files = sorted(glob(os.path.join(nc_dir, '*.nc')))

    weather_feature_cols = ['ghi_mean', 'ghi_night']

    weather_dfs = []
    print(f"开始加载 {len(nc_files)} 个气象文件（仅保留 ghi_mean/ghi_night）...")
    for idx, f in enumerate(nc_files, start=1):
        try:
            wdf = load_and_process_nc(f)
            weather_dfs.append(wdf)
            if idx % 50 == 0 or idx == 1:
                print(f"  已处理 {idx}/{len(nc_files)} 个文件...")
        except Exception as e:
            print(f"跳过文件 {f}: {e}")
    print("气象文件加载完成！")

    if weather_dfs:
        weather_all = pd.concat(weather_dfs, ignore_index=True)
        df_train['date'] = df_train['times'].dt.date
        df_train = pd.merge(df_train, weather_all, on='date', how='left')
        df_train = df_train.drop(columns=['date'])
    else:
        for col in weather_feature_cols:
            df_train[col] = 0.0

    df_train = add_time_features(df_train)
    all_features = feature_cols + ['hour', 'dayofweek', 'month'] + weather_feature_cols

    # 填充缺失值
    df_train[weather_feature_cols] = df_train[weather_feature_cols].ffill().fillna(0)
    if df_train[all_features].isnull().any().any():
        df_train[all_features] = df_train[all_features].ffill().fillna(0)

    X = df_train[all_features].values
    y = df_train[target_col].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # 2. 模型训练
    print("\n[2/4] 训练模型...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        verbose=1
    )
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    print(f'\n  验证集 RMSE: {rmse:.6f}, MAE: {mae:.6f}')

    print("\n特征重要性排序(从高到低):")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(all_features, importance), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    # 3. 测试集推理
    print("\n[3/4] 测试集推理...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])

    weather_test_dfs = []
    for f in nc_files:
        try:
            wdf = load_and_process_nc(f)
            weather_test_dfs.append(wdf)
        except Exception:
            pass

    if weather_test_dfs:
        weather_test_all = pd.concat(weather_test_dfs, ignore_index=True)
        df_test['date'] = df_test['times'].dt.date
        df_test = pd.merge(df_test, weather_test_all, on='date', how='left')
        df_test = df_test.drop(columns=['date'])
        df_test[weather_feature_cols] = df_test[weather_feature_cols].ffill().fillna(0)
    else:
        for col in weather_feature_cols:
            df_test[col] = 0.0

    df_test = add_time_features(df_test)

    if df_test[all_features].isnull().any().any():
        df_test[all_features] = df_test[all_features].ffill().fillna(0)

    X_test = df_test[all_features].values
    y_test_pred = model.predict(X_test)

    df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
    df_out.to_csv(output_price_path, index=False)
    print(f'  推理结果已保存: {output_price_path}')

    # 4. 生成充放电策略（如需加阈值，修改下面这行：min_profit_threshold=1250）
    print("\n[4/4] 生成充放电策略...")
    generate_strategy(output_price_path, output_power_path, min_profit_threshold=2000)

    print("\n" + "=" * 60)
    print("运行完成！提交文件:", output_power_path)
    print("=" * 60)