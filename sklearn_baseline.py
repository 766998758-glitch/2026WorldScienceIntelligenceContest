"""
Sklearn GradientBoosting 基线：根据边界条件预测节点电价 A
（纯sklearn实现，无需LightGBM/XGBoost）
"""
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 路径配置 ====================
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 数据目录（请根据实际下载的数据位置修改）
data_dir = os.path.join(current_dir, 'data')
train_feature_path = os.path.join(data_dir, 'train', 'mengxi_boundary_anon_filtered.csv')
train_label_path = os.path.join(data_dir, 'train', 'mengxi_node_price_selected.csv')
test_feature_path = os.path.join(data_dir, 'test', 'test_in_feature_ori.csv')

# 输出目录
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
output_price_path = os.path.join(output_dir, 'sklearn_baseline_output.csv')
output_power_path = os.path.join(output_dir, 'output.csv')  # 最终提交文件

# 边界条件特征列（与测试集对齐，仅使用预测值列）
feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']
target_col = 'A'


# 添加时间特征
def add_time_features(df):
    df = df.copy()
    df['hour'] = df['times'].dt.hour
    df['minute'] = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month'] = df['times'].dt.month
    return df

def load_and_process_nc(filepath):
    import xarray as xr

    ds = xr.open_dataset(filepath)

    u100 = ds['data'].sel(channel='u100')
    v100 = ds['data'].sel(channel='v100')
    ghi  = ds['data'].sel(channel='ghi')

    ws = np.sqrt(u100 ** 2 + v100 ** 2)

    ws_mean = ws.mean(dim=['lat', 'lon'])
    ghi_mean = ghi.mean(dim=['lat', 'lon'])

    ws_hourly = ws_mean.values.flatten()
    ghi_hourly = ghi_mean.values.flatten()

    pub_date = pd.to_datetime(ds['time'].values[0])
    target_date = pub_date + pd.Timedelta(days=1)

    hours_orig = np.arange(24)
    minutes_new = np.arange(96) * 15 / 60
    ws_15min = np.interp(minutes_new, hours_orig, ws_hourly)
    ghi_15min = np.interp(minutes_new, hours_orig, ghi_hourly)

    # 注意：去掉了 tz 参数
    times = pd.date_range(
        start=target_date,
        periods=96,
        freq='15min'
    )

    ds.close()
    return pd.DataFrame({
        'times': times,
        'wind_speed': ws_15min,
        'ghi': ghi_15min
    })

# ==================== 充放电策略生成 ====================
def generate_strategy(price_csv, save_path):
    """
    根据预测的实时价格确定充放电策略
    
    策略逻辑：
    1. 寻找最优的充电开始时间tc和放电开始时间td
    2. 充电持续8个时间点（2小时），放电持续8个时间点（2小时）
    3. 充电开始时间：0 <= tc <= 80
    4. 放电开始时间：td >= tc + 8 且 td <= 88
    5. 目标：最大化收益 = sum(放电时段价格) * 1000 - sum(充电时段价格) * 1000
    """
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])
    
    # 按天分组处理
    df['date'] = df['times'].dt.date
    
    results = []
    total_profit = 0
    
    for date, group in df.groupby('date'):
        prices = group['A'].values
        times = group['times'].values
        
        n = len(prices)
        if n != 96:
            print(f"警告: {date} 的数据点数量为 {n}, 预期为 96")
            continue
        
        best_profit = 0
        best_tc = -1
        best_td = -1

        # 遍历所有可能的充电和放电开始时间
        for tc in range(0, 81):  # 0 <= tc <= 80
            # 充电时段价格总和
            charge_prices = prices[tc:tc+8]
            charge_cost = np.sum(charge_prices) * 1000  # 充电成本

            for td in range(tc + 8, 89):  # td >= tc + 8 且 td <= 88
                # 放电时段价格总和
                discharge_prices = prices[td:td+8]
                discharge_revenue = np.sum(discharge_prices) * 1000  # 放电收入

                # 计算收益
                profit = discharge_revenue - charge_cost

                if profit > best_profit:
                    best_profit = profit
                    best_tc = tc
                    best_td = td
        
        # 生成充放电策略
        power = np.zeros(96)
        if best_tc >= 0 and best_td >= 0:
            # 充电：-1000
            power[best_tc:best_tc+8] = -1000
            # 放电：+1000
            power[best_td:best_td+8] = 1000
            total_profit += best_profit
            print(f"日期: {date}, 充电开始: {best_tc:2d}, 放电开始: {best_td:2d}, 预期收益: {best_profit:10.2f}")
        else:
            print(f"日期: {date}, 无交易（收益非正）")
        
        # 构建结果
        for i, (t, p, pr) in enumerate(zip(times, power, prices)):
            results.append({
                'times': t,
                '实时价格': pr,
                'power': p
            })
    
    # 保存结果
    df_result = pd.DataFrame(results)
    df_result.to_csv(save_path, index=False)
    
    n_days = len(df.groupby("date"))
    avg_profit = total_profit / n_days if n_days > 0 else 0
    
    print(f'\n充放电策略已保存: {save_path}')
    print(f'总天数: {n_days}')
    print(f'总收益: {total_profit:.2f}')
    print(f'平均日收益: {avg_profit:.2f}')
    
    return df_result


# ==================== 主程序 ====================
if __name__ == '__main__':
    # 检查数据文件是否存在
    if not os.path.exists(train_feature_path):
        print(f"错误: 训练特征文件不存在: {train_feature_path}")
        print("请从赛事官网下载数据并放置到正确的位置")
        print("数据目录结构应为:")
        print("  data/")
        print("    train/")
        print("      mengxi_boundary_anon_filtered.csv")
        print("      mengxi_node_price_selected.csv")
        print("    test/")
        print("      test_in_feature_ori.csv")
        exit(1)
    
    if not os.path.exists(train_label_path):
        print(f"错误: 训练标签文件不存在: {train_label_path}")
        exit(1)
    
    if not os.path.exists(test_feature_path):
        print(f"错误: 测试特征文件不存在: {test_feature_path}")
        exit(1)
    
    print("=" * 60)
    print("开始训练 Sklearn GradientBoosting 模型...")
    print("=" * 60)
    
    # ==================== 1. 数据准备 ====================
    print("\n[1/4] 加载数据...")
    df_feat = pd.read_csv(train_feature_path)
    df_label = pd.read_csv(train_label_path)
    print(f"  训练特征: {df_feat.shape}")
    print(f"  训练标签: {df_label.shape}")

    # 按 times 内连接对齐
    df_train = pd.merge(df_feat, df_label, on='times', how='inner')
    df_train['times'] = pd.to_datetime(df_train['times'])
    print(f"  合并后: {df_train.shape}")
    
    # ------- 新增：加载气象数据并合并 -------
    import xarray as xr
    from glob import glob

    # 注意路径改成你刚才验证通过的路径
    nc_dir = '/mnt/workspace/all_nc/to_sais_new/all_nc'   # 或者用绝对路径
    nc_files = sorted(glob(os.path.join(nc_dir, '*.nc')))

    weather_dfs = []
    print(f"开始加载 {len(nc_files)} 个气象文件...")
    for idx, f in enumerate(nc_files, start=1):  # start=1 让计数从1开始更直观
        try:
            wdf = load_and_process_nc(f)
            weather_dfs.append(wdf)
            # 每处理 50 个文件，或处理第 1 个文件时，输出进度
            if idx % 50 == 0 or idx == 1:
                print(f"  已处理 {idx}/{len(nc_files)} 个文件...")
        except Exception as e:
            print(f"跳过文件 {f}: {e}")
    print("气象文件加载完成！")

    if weather_dfs:
        weather_all = pd.concat(weather_dfs, ignore_index=True)
        # 按 times 左连接（保留 df_train 所有行，气象缺失则为 NaN）
        df_train = pd.merge(df_train, weather_all, on='times', how='left')
        print("气象特征已合并，新增列：", [c for c in weather_all.columns if c != 'times'])
    else:
        print("警告：未加载任何气象数据，请检查路径")
    # --------------------------------------------

    df_train = add_time_features(df_train)
    all_features = feature_cols + ['hour', 'dayofweek', 'month', 'wind_speed', 'ghi']
    # 填充气象特征的缺失值
    df_train[['wind_speed', 'ghi']] = df_train[['wind_speed', 'ghi']].ffill()
    
    # === 新增：确保万无一失 ===
    # 1. 检查是否还有 NaN
    if df_train[['wind_speed', 'ghi']].isnull().any().any():
        print("⚠️ 气象特征仍有缺失值，尝试用 0 填充...")
        df_train[['wind_speed', 'ghi']] = df_train[['wind_speed', 'ghi']].fillna(0)
    
    # 2. 检查整个特征集
    df_check = df_train[all_features]
    if df_check.isnull().any().any():
        print("⚠️ 特征矩阵仍存在 NaN，将再次填充（前向/0）...")
        df_train[all_features] = df_train[all_features].ffill().fillna(0)
    # =======================
    
    X = df_train[all_features].values
    y = df_train[target_col].values


    # 按时间顺序划分，最后20%做验证
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # ==================== 2. 模型训练 ====================
    print("\n[2/4] 训练模型...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    # 验证集评估
    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    print(f'\n  验证集 RMSE: {rmse:.6f}, MAE: {mae:.6f}')
    
    # ===================== 看特征重要性 =====================
    print("\n特征重要性排序(从高到低):")
    importance = model.feature_importances_
    for feat, imp in sorted(zip(all_features, importance), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    

    # ==================== 3. 测试集推理 ====================
    print("\n[3/4] 测试集推理...")
    df_test = pd.read_csv(test_feature_path)
    df_test['times'] = pd.to_datetime(df_test['times'])
    
    # 加载气象数据（复用刚才的 nc_files 即可）
    # 如果之前已经加载了 weather_all，可以直接 merge（要保证 weather_all 包含测试集所有日期）
    # 简单方式：直接重新生成一份 weather_all，然后 merge
    weather_test_dfs = []
    for f in nc_files:
        try:
            wdf = load_and_process_nc(f)
            weather_test_dfs.append(wdf)
        except Exception as e:
            pass

    if weather_test_dfs:
        weather_test_all = pd.concat(weather_test_dfs, ignore_index=True)
        df_test = pd.merge(df_test, weather_test_all, on='times', how='left')
        df_test[['wind_speed', 'ghi']] = df_test[['wind_speed', 'ghi']].ffill()
    df_test = add_time_features(df_test)
    
    # === 新增：确保万无一失 ===
    # 1. 检查是否还有 NaN
    if df_test[['wind_speed', 'ghi']].isnull().any().any():
        print("⚠️ 气象特征仍有缺失值，尝试用 0 填充...")
        df_test[['wind_speed', 'ghi']] = df_test[['wind_speed', 'ghi']].fillna(0)
    
    # 2. 检查整个特征集
    df_check = df_train[all_features]
    if df_check.isnull().any().any():
        print("⚠️ 特征矩阵仍存在 NaN，将再次填充（前向/0）...")
        df_test[all_features] = df_test[all_features].ffill().fillna(0)
    # =======================

    X_test = df_test[all_features].values
    y_test_pred = model.predict(X_test)

    df_out = pd.DataFrame({'times': df_test['times'], target_col: y_test_pred})
    df_out.to_csv(output_price_path, index=False)
    print(f'  推理结果已保存: {output_price_path}')
    print(f'  预测天数: {len(df_out) // 96} 天')
    
    # ==================== 4. 生成充放电策略 ====================
    print("\n[4/4] 生成充放电策略...")
    generate_strategy(output_price_path, output_power_path)
    
    print("\n" + "=" * 60)
    print("Baseline运行完成！")
    print(f"提交文件: {output_power_path}")
    print("=" * 60)