"""
LightGBM 两步走基线 + 气象特征 + 网格搜索最优参数
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from glob import glob

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

# 边界条件特征列（仅用预测值列）
feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
target_col = 'A'

# ==================== 时间特征 ====================
def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df

# ==================== 气象特征处理 ====================
def load_and_process_nc(filepath):
    """读取单个 nc 文件，返回日度聚合气象特征（10个统计量）"""
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

    ws_mean = np.mean(ws_hourly)
    ghi_mean = np.mean(ghi_hourly)
    ws_max = np.max(ws_hourly)
    ws_min = np.min(ws_hourly)
    ghi_max = np.max(ghi_hourly)
    ghi_min = np.min(ghi_hourly)
    ws_afternoon = np.mean(ws_hourly[12:18]) if len(ws_hourly) >= 18 else ws_mean
    ghi_afternoon = np.mean(ghi_hourly[12:18]) if len(ghi_hourly) >= 18 else ghi_mean
    ws_night = np.mean(ws_hourly[0:6])
    ghi_night = np.mean(ghi_hourly[0:6])

    pub_date = pd.to_datetime(str(ds['time'].values[0]))
    target_date = pd.Timestamp(pub_date + pd.Timedelta(days=1)).date()
    ds.close()

    return pd.DataFrame({
        'date': [target_date],
        'ws_mean': ws_mean, 'ws_max': ws_max, 'ws_min': ws_min,
        'ws_afternoon': ws_afternoon, 'ws_night': ws_night,
        'ghi_mean': ghi_mean, 'ghi_max': ghi_max,
        'ghi_afternoon': ghi_afternoon, 'ghi_night': ghi_night
    })

# ==================== 充放电策略生成（无阈值版本，可后续加入） ====================
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
        best_tc, best_td = -1, -1
        for tc in range(0, 81):
            charge_cost = np.sum(prices[tc:tc+8]) * 1000
            for td in range(tc + 8, 89):
                discharge_revenue = np.sum(prices[td:td+8]) * 1000
                profit = discharge_revenue - charge_cost
                if profit > best_profit:
                    best_profit = profit
                    best_tc, best_td = tc, td

        power = np.zeros(96)
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
    # 检查数据文件
    for path in [train_feature_path, train_label_path, test_feature_path]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在: {path}")
            exit(1)

    print("=" * 60)
    print("LightGBM + 气象特征 + 网格搜索")
    print("=" * 60)

    # ==================== 1. 数据准备 ====================
    print("\n[1/4] 加载数据...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)
    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])

    # ------- 加载气象特征并合并 -------
    print("加载气象数据...")
    nc_dir = '/mnt/workspace/all_nc/to_sais_new/all_nc'   # 【需调整】你的气象文件路径
    nc_files = sorted(glob(os.path.join(nc_dir, '*.nc')))
    weather_feature_cols = [
        'ws_mean', 'ws_max', 'ws_min', 'ws_afternoon', 'ws_night',
        'ghi_mean', 'ghi_max', 'ghi_afternoon', 'ghi_night'
    ]

    weather_dfs = []
    for idx, f in enumerate(nc_files, start=1):
        try:
            wdf = load_and_process_nc(f)
            weather_dfs.append(wdf)
            if idx % 50 == 0 or idx == 1:
                print(f"  已处理 {idx}/{len(nc_files)} 个气象文件...")
        except Exception as e:
            print(f"跳过 {f}: {e}")
    print("气象文件加载完成！")

    if weather_dfs:
        weather_all = pd.concat(weather_dfs, ignore_index=True)
        df_train['date'] = df_train['times'].dt.date
        df_train = pd.merge(df_train, weather_all, on='date', how='left')
        df_train.drop(columns=['date'], inplace=True)
    else:
        for col in weather_feature_cols:
            df_train[col] = 0.0

    # 添加时间特征
    df_train = add_time_features(df_train)
    all_features = feature_cols + ['hour', 'dayofweek', 'month'] + weather_feature_cols

    # 填充气象缺失值
    df_train[weather_feature_cols] = df_train[weather_feature_cols].ffill().fillna(0)

    # 划分 X, y
    X = df_train[all_features].values
    y = df_train[target_col].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # ==================== 2. 模型训练（网格搜索） ====================
    print("\n[2/4] 网格搜索最优参数（基于验证集 RMSE）...")
    best_rmse = np.inf
    best_params = {}
    best_model = None

    # 搜索范围 【需调整】
    param_grid = {
        'reg_alpha': [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06],
        'reg_lambda': [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06],
        'max_depth': [4, 5, 6, 7],
        'early_stopping_rounds': [None]   # 关闭早停
    }

    for reg_alpha in param_grid['reg_alpha']:
        for reg_lambda in param_grid['reg_lambda']:
            for max_depth in param_grid['max_depth']:
                for es_rounds in param_grid['early_stopping_rounds']:
                    n_estimators = 200 if es_rounds is None else 1000
                    use_early_stop = es_rounds is not None
                    desc = "早停关" if es_rounds is None else f"早停({es_rounds}轮)"

                    print(f"  测试 alpha={reg_alpha:.3f}, lambda={reg_lambda:.3f}, depth={max_depth}, {desc} ...", end=" ")

                    model = lgb.LGBMRegressor(
                        n_estimators=n_estimators,
                        learning_rate=0.05,
                        max_depth=max_depth,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1
                    )

                    if use_early_stop:
                        model.fit(X_train, y_train,
                                  eval_set=[(X_val, y_val)],
                                  eval_metric='rmse',
                                  callbacks=[lgb.early_stopping(stopping_rounds=es_rounds),
                                             lgb.log_evaluation(0)])
                    else:
                        model.fit(X_train, y_train)

                    y_val_pred = model.predict(X_val)
                    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
                    mae_val = mean_absolute_error(y_val, y_val_pred)
                    print(f"RMSE={rmse_val:.6f}, MAE={mae_val:.6f}")

                    if rmse_val < best_rmse:
                        best_rmse = rmse_val
                        best_params = {
                            'reg_alpha': reg_alpha,
                            'reg_lambda': reg_lambda,
                            'max_depth': max_depth,
                            'early_stopping_rounds': es_rounds
                        }
                        best_model = model

    print(f"\n最优参数: {best_params}")
    print(f"最优验证集 RMSE: {best_rmse:.6f}")

    # 使用最优模型
    model = best_model

    # 重新评估并输出特征重要性
    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    print(f'\n  验证集 RMSE: {rmse:.6f}, MAE: {mae:.6f}')

    print("\n特征重要性排序(从高到低):")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(all_features, importance), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    # ==================== 3. 测试集推理 ====================
    print("\n[3/4] 测试集推理...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    df_test = add_time_features(df_test)

    # 同样为测试集加载气象特征
    weather_test_dfs = []
    for f in nc_files:
        try:
            weather_test_dfs.append(load_and_process_nc(f))
        except:
            pass
    if weather_test_dfs:
        weather_test_all = pd.concat(weather_test_dfs, ignore_index=True)
        df_test['date'] = df_test['times'].dt.date
        df_test = pd.merge(df_test, weather_test_all, on='date', how='left')
        df_test.drop(columns=['date'], inplace=True)
        df_test[weather_feature_cols] = df_test[weather_feature_cols].ffill().fillna(0)
    else:
        for col in weather_feature_cols:
            df_test[col] = 0.0

    X_test = df_test[all_features].values
    y_test_pred = model.predict(X_test)

    df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
    df_out.to_csv(output_price_path, index=False)
    print(f'  推理结果已保存: {output_price_path}')

    # ==================== 4. 生成充放电策略 ====================
    print("\n[4/4] 生成充放电策略...")
    generate_strategy(output_price_path, output_power_path, min_profit_threshold=0)  # 【可调整阈值】

    print("\n" + "=" * 60)
    print("运行完成！提交文件:", output_power_path)
    print("=" * 60)