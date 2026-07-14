"""
纽约黄色出租车 Parquet 数据处理脚本（字段兼容修复版）
修复点：所有字段操作前先校验是否存在，避免 KeyError
"""

import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq

# 创建输出目录
os.makedirs('outputs', exist_ok=True)

# --------------------------
# 1. 加载 Parquet 数据 + 生成数据质量报告
# --------------------------
file_path = 'yellow_tripdata_2026-01.parquet'
table = pq.read_table(file_path)
df_raw = table.to_pandas()

# 打印实际字段名，方便排查
print("数据实际包含字段：")
print(df_raw.columns.tolist())
print("-" * 50)

def generate_data_quality_report(data: pd.DataFrame) -> pd.DataFrame:
    """
    生成数据质量报告
    """
    report = pd.DataFrame({
        '字段名称': data.columns,
        '数据类型': data.dtypes.astype(str).values,
        '总记录数': len(data),
        '非空记录数': data.notnull().sum().values,
        '缺失记录数': data.isnull().sum().values,
        '缺失率(%)': (data.isnull().sum() / len(data) * 100).round(4).values
    })

    # 数值型字段异常值统计
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    report['负值数量'] = 0
    report['零值数量'] = 0
    report['99分位极端值数量'] = 0

    for col in numeric_cols:
        idx = report[report['字段名称'] == col].index[0]
        report.loc[idx, '负值数量'] = int((data[col] < 0).sum())
        report.loc[idx, '零值数量'] = int((data[col] == 0).sum())
        q99 = data[col].quantile(0.99)
        report.loc[idx, '99分位极端值数量'] = int((data[col] > q99).sum())

    # 行程时间逻辑异常校验
    if {'tpep_pickup_datetime', 'tpep_dropoff_datetime'}.issubset(data.columns):
        invalid_time_count = (data['tpep_dropoff_datetime'] < data['tpep_pickup_datetime']).sum()
        logic_error_row = pd.DataFrame([{
            '字段名称': '行程时间逻辑异常(下车早于上车)',
            '数据类型': '逻辑校验',
            '总记录数': len(data),
            '非空记录数': '-',
            '缺失记录数': '-',
            '缺失率(%)': '-',
            '负值数量': '-',
            '零值数量': '-',
            '99分位极端值数量': int(invalid_time_count)
        }])
        report = pd.concat([report, logic_error_row], ignore_index=True)

    return report

# 生成并保存质量报告
quality_report = generate_data_quality_report(df_raw)
quality_report.to_csv('outputs/data_quality_report.csv', index=False, encoding='utf-8-sig')
print("✅ 数据质量报告已保存：outputs/data_quality_report.csv")
print(f"原始数据总行数：{len(df_raw)}，字段数：{len(df_raw.columns)}")

# --------------------------
# 2. 数据清洗策略（字段存在性校验修复）
# --------------------------
df_clean = df_raw.copy()

# ========== 修复核心：只取数据中真实存在的字段 ==========
# 费用类字段候选列表（包含可能存在的所有费用字段）
fee_candidates = [
    'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
    'improvement_surcharge', 'congestion_surcharge',
    'airport_fee', 'cbd_congestion_fee'
]
# 过滤出数据中真实存在的字段
fee_cols = [col for col in fee_candidates if col in df_clean.columns]

# 清洗策略1：费用类字段缺失值填充为 0
# 理由：缺失代表该行程未产生对应费用，填充0符合业务逻辑
if fee_cols:
    df_clean[fee_cols] = df_clean[fee_cols].fillna(0)

# 清洗策略2：删除核心字段缺失的记录
core_candidates = [
    'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
    'PULocationID', 'DOLocationID', 'fare_amount', 'total_amount'
]
core_cols = [col for col in core_candidates if col in df_clean.columns]
df_clean = df_clean.dropna(subset=core_cols)

# 清洗策略3：过滤行程时间逻辑异常记录
if 'tpep_pickup_datetime' in df_clean.columns and 'tpep_dropoff_datetime' in df_clean.columns:
    df_clean = df_clean[df_clean['tpep_dropoff_datetime'] > df_clean['tpep_pickup_datetime']]

# 清洗策略4：过滤数值字段的负值
non_negative_candidates = [
    'trip_distance', 'fare_amount', 'extra', 'mta_tax',
    'tip_amount', 'tolls_amount', 'improvement_surcharge',
    'total_amount', 'congestion_surcharge', 'airport_fee', 'cbd_congestion_fee'
]
non_negative_cols = [col for col in non_negative_candidates if col in df_clean.columns]
for col in non_negative_cols:
    df_clean = df_clean[df_clean[col] >= 0]

# 清洗策略5：校验分类字段的合法枚举值
if 'VendorID' in df_clean.columns:
    df_clean = df_clean[df_clean['VendorID'].isin([1, 2, 6, 7])]
if 'RatecodeID' in df_clean.columns:
    df_clean = df_clean[df_clean['RatecodeID'].isin([1,2,3,4,5,6,99])]
if 'payment_type' in df_clean.columns:
    df_clean = df_clean[df_clean['payment_type'].isin([0,1,2,3,4,5,6])]
if 'store_and_fwd_flag' in df_clean.columns:
    df_clean['store_and_fwd_flag'] = df_clean['store_and_fwd_flag'].fillna('N')
    df_clean = df_clean[df_clean['store_and_fwd_flag'].isin(['Y', 'N'])]

# 清洗策略6：过滤业务不合理的极端值
if 'trip_distance' in df_clean.columns:
    df_clean = df_clean[df_clean['trip_distance'] <= 100]
    # 距离为0但产生基础车费，不符合计价逻辑
    if 'fare_amount' in df_clean.columns:
        df_clean = df_clean[~((df_clean['trip_distance'] == 0) & (df_clean['fare_amount'] > 0))]

# 乘客数处理
if 'passenger_count' in df_clean.columns:
    mode_val = df_clean['passenger_count'].mode()[0] if not df_clean['passenger_count'].mode().empty else 1
    df_clean['passenger_count'] = df_clean['passenger_count'].fillna(mode_val)
    df_clean = df_clean[(df_clean['passenger_count'] >= 1) & (df_clean['passenger_count'] <= 6)]

print(f"🧹 数据清洗完成，清洗后行数：{len(df_clean)}，剔除异常记录：{len(df_raw) - len(df_clean)} 条")

# --------------------------
# 3. 提取基础时间特征
# --------------------------
if 'tpep_pickup_datetime' in df_clean.columns and 'tpep_dropoff_datetime' in df_clean.columns:
    df_clean['tpep_pickup_datetime'] = pd.to_datetime(df_clean['tpep_pickup_datetime'])
    df_clean['tpep_dropoff_datetime'] = pd.to_datetime(df_clean['tpep_dropoff_datetime'])

    df_clean['pickup_hour'] = df_clean['tpep_pickup_datetime'].dt.hour
    df_clean['pickup_weekday'] = df_clean['tpep_pickup_datetime'].dt.weekday
    df_clean['is_weekend'] = (df_clean['pickup_weekday'] >= 5).astype(int)

    # 高峰时段：工作日早7-9、晚17-19
    is_workday = df_clean['is_weekend'] == 0
    morning_rush = (df_clean['pickup_hour'] >= 7) & (df_clean['pickup_hour'] < 9)
    evening_rush = (df_clean['pickup_hour'] >= 17) & (df_clean['pickup_hour'] < 19)
    df_clean['is_rush_hour'] = (is_workday & (morning_rush | evening_rush)).astype(int)

# --------------------------
# 4. 构造衍生特征
# --------------------------
# 衍生特征1：行程时长（分钟）
if 'tpep_pickup_datetime' in df_clean.columns and 'tpep_dropoff_datetime' in df_clean.columns:
    df_clean['trip_duration_min'] = (
        df_clean['tpep_dropoff_datetime'] - df_clean['tpep_pickup_datetime']
    ).dt.total_seconds() / 60
    # 过滤小于1分钟的异常行程
    df_clean = df_clean[df_clean['trip_duration_min'] >= 1]

    # 衍生特征2：平均行驶速度（英里/小时）
    if 'trip_distance' in df_clean.columns:
        df_clean['avg_speed_mph'] = df_clean['trip_distance'] / (df_clean['trip_duration_min'] / 60)

# 衍生特征3：单位里程总费用（美元/英里）
if 'total_amount' in df_clean.columns and 'trip_distance' in df_clean.columns:
    df_clean['cost_per_mile'] = np.where(
        df_clean['trip_distance'] > 0,
        df_clean['total_amount'] / df_clean['trip_distance'],
        0
    )

print("\n✨ 特征工程完成，最终数据集字段列表：")
print(df_clean.columns.tolist())

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# 全局配置：中文字体与图表风格
# --------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'Microsoft YaHei UI',
    'SimSun', 'Arial Unicode MS'
]
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {"font.sans-serif": plt.rcParams['font.sans-serif']})

# --------------------------
# 1. 出行需求时间规律分析
# --------------------------
hour_demand = df_clean.groupby(['pickup_hour', 'is_weekend']).size().reset_index(name='order_count')
hour_demand['时段类型'] = hour_demand['is_weekend'].map({0: '工作日', 1: '周末'})

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=hour_demand,
    x='pickup_hour',
    y='order_count',
    hue='时段类型',
    marker='o',
    linewidth=2,
    palette=['#1f77b4', '#ff7f0e']
)
plt.title('工作日与周末分小时出行订单量对比', fontsize=14, pad=15)
plt.xlabel('小时（0-23）', fontsize=12)
plt.ylabel('订单数量', fontsize=12)
plt.xticks(range(0, 24))
plt.legend(title='时段类型', frameon=True)
plt.tight_layout()
plt.savefig('outputs/m2_1_time_demand.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存图表：outputs/m2_1_time_demand.png")

# --------------------------
# 2. 区域热度分析
# --------------------------
pu_top10 = df_clean['PULocationID'].value_counts().head(10).sort_values(ascending=True)
do_top10 = df_clean['DOLocationID'].value_counts().head(10).sort_values(ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.barh(pu_top10.index.astype(str), pu_top10.values, color='#1f77b4')
ax1.set_title('上车订单量TOP10区域', fontsize=13, pad=10)
ax1.set_xlabel('订单数量', fontsize=11)
ax1.set_ylabel('TLC出租车区域ID', fontsize=11)

ax2.barh(do_top10.index.astype(str), do_top10.values, color='#ff7f0e')
ax2.set_title('下车订单量TOP10区域', fontsize=13, pad=10)
ax2.set_xlabel('订单数量', fontsize=11)
ax2.set_ylabel('TLC出租车区域ID', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/m2_2_top10_zone.png', dpi=300, bbox_inches='tight')
plt.close()

# 热力图
top10_pu_zones = pu_top10.index.tolist()
zone_hour_pivot = df_clean[df_clean['PULocationID'].isin(top10_pu_zones)] \
    .groupby(['PULocationID', 'pickup_hour']).size().unstack(fill_value=0)

plt.figure(figsize=(14, 8))
sns.heatmap(zone_hour_pivot, cmap='YlOrRd', annot=False, cbar_kws={'label': '订单数量'})
plt.title('TOP10上车区域分小时订单热力图', fontsize=14, pad=15)
plt.xlabel('小时（0-23）', fontsize=12)
plt.ylabel('TLC出租车区域ID', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/m2_2_zone_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存图表：outputs/m2_2_top10_zone.png、outputs/m2_2_zone_heatmap.png")

# --------------------------
# 3. 车费影响因素分析
# --------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1：距离-车费散点图
sample_df = df_clean.sample(n=10000, random_state=42)
sns.scatterplot(
    data=sample_df,
    x='trip_distance',
    y='total_amount',
    alpha=0.4,
    s=10,
    color='#2ca02c',
    ax=axes[0, 0]
)
axes[0, 0].set_title('行程距离与总车费散点图（抽样1万条）', fontsize=13, pad=10)
axes[0, 0].set_xlabel('行程距离（英里）', fontsize=11)
axes[0, 0].set_ylabel('总车费（美元）', fontsize=11)

# 子图2：高峰vs非高峰基础车费箱线图
df_clean['高峰标识'] = df_clean['is_rush_hour'].map({0: '非高峰', 1: '高峰'})
sns.boxplot(
    data=df_clean,
    x='高峰标识',
    y='fare_amount',
    hue='高峰标识',
    showfliers=False,
    palette=['#9467bd', '#8c564b'],
    legend=False,
    ax=axes[0, 1]
)
axes[0, 1].set_title('高峰与非高峰时段基础车费分布', fontsize=13, pad=10)
axes[0, 1].set_xlabel('时段类型', fontsize=11)
axes[0, 1].set_ylabel('基础车费（美元）', fontsize=11)

# 子图3：乘客人数-平均车费柱状图
passenger_avg_fare = df_clean.groupby('passenger_count')['total_amount'].mean().reset_index()
sns.barplot(
    data=passenger_avg_fare,
    x='passenger_count',
    y='total_amount',
    hue='passenger_count',
    palette='Blues_d',
    legend=False,
    ax=axes[1, 0]
)
axes[1, 0].set_title('不同乘客人数的平均总车费', fontsize=13, pad=10)
axes[1, 0].set_xlabel('乘客人数', fontsize=11)
axes[1, 0].set_ylabel('平均总车费（美元）', fontsize=11)

# 子图4：工作日vs周末车费小提琴图
df_clean['时段类型'] = df_clean['is_weekend'].map({0: '工作日', 1: '周末'})
sns.violinplot(
    data=df_clean,
    x='时段类型',
    y='total_amount',
    hue='时段类型',
    palette=['#1f77b4', '#ff7f0e'],
    legend=False,
    ax=axes[1, 1]
)
axes[1, 1].set_title('工作日与周末总车费分布对比', fontsize=13, pad=10)
axes[1, 1].set_xlabel('时段类型', fontsize=11)
axes[1, 1].set_ylabel('总车费（美元）', fontsize=11)
axes[1, 1].set_ylim(0, df_clean['total_amount'].quantile(0.99))

plt.tight_layout()
plt.savefig('outputs/m2_3_fare_factors.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存图表：outputs/m2_3_fare_factors.png")

# --------------------------
# 4. 自选分析：城市交通拥堵时空特征分析
# 修复：统一字段名为 avg_speed_mph，与特征工程生成的字段一致
# --------------------------
# 分小时平均车速（工作日vs周末）
hour_speed = df_clean.groupby(['pickup_hour', 'is_weekend'])['avg_speed_mph'].mean().reset_index()
hour_speed['时段类型'] = hour_speed['is_weekend'].map({0: '工作日', 1: '周末'})

# 区域平均车速统计
zone_speed_stats = df_clean.groupby('PULocationID').agg(
    avg_speed_mph=('avg_speed_mph', 'mean'),
    order_count=('PULocationID', 'count')
).reset_index()
zone_speed_stats = zone_speed_stats[zone_speed_stats['order_count'] >= 100]
# 取车速最低的TOP10拥堵区域
congestion_top10 = zone_speed_stats.sort_values('avg_speed_mph', ascending=True).head(10) \
    .sort_values('avg_speed_mph', ascending=False)

# 绘制双子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

sns.lineplot(
    data=hour_speed,
    x='pickup_hour',
    y='avg_speed_mph',
    hue='时段类型',
    marker='o',
    linewidth=2,
    palette=['#1f77b4', '#2ca02c'],
    ax=ax1
)
ax1.set_title('工作日与周末分小时平均行驶速度对比', fontsize=13, pad=10)
ax1.set_xlabel('小时（0-23）', fontsize=11)
ax1.set_ylabel('平均行驶速度（英里/小时）', fontsize=11)
ax1.set_xticks(range(0, 24))
ax1.legend(title='时段类型', frameon=True)

ax2.barh(congestion_top10['PULocationID'].astype(str), congestion_top10['avg_speed_mph'], color='#d62728')
ax2.set_title('拥堵程度TOP10区域（平均车速最低）', fontsize=13, pad=10)
ax2.set_xlabel('平均行驶速度（英里/小时）', fontsize=11)
ax2.set_ylabel('TLC出租车区域ID', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/m2_4_congestion_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ 已保存图表：outputs/m2_4_congestion_analysis.png")

print("\n🎉 所有分析图表已全部生成并保存至 outputs/ 目录")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# 全局随机种子固定，保证实验完全可复现
# --------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------
# 构造出行需求量预测数据集
# 样本粒度：某区域（PULocationID）某一天某一小时
# 目标变量：该时段该区域的订单数量（出行需求量）
# --------------------------
# 提取日期维度，用于按天+小时聚合订单量
df_clean['pickup_date'] = df_clean['tpep_pickup_datetime'].dt.date

# 按区域、日期、小时三维度聚合，构造每条样本
demand_data = df_clean.groupby(['PULocationID', 'pickup_date', 'pickup_hour']).agg(
    demand=('VendorID', 'count'),  # 目标变量：时段订单需求量
    pickup_weekday=('pickup_weekday', 'first'),
    is_weekend=('is_weekend', 'first'),
    is_rush_hour=('is_rush_hour', 'first')
).reset_index()

# --------------------------
# 输入特征设计与理由说明
# 1. PULocationID：区域标识。不同区域的功能定位（商业区/住宅区/机场/景区）、
#    人口密度差异极大，是决定基础需求量的核心因素，必须作为核心输入特征。
# 2. pickup_hour：小时维度。日内出行需求呈现极强的时段规律，早晚通勤高峰、
#    午间平峰、夜间低谷模式稳定，是预测需求量的关键时间特征。
# 3. pickup_weekday：星期维度。周一到周日的出行需求有明显差异，工作日以通勤为主，
#    周末以休闲出行为主，不同星期的需求曲线形态不同。
# 4. is_weekend：是否周末。作为星期维度的高阶业务抽象，直接区分工作日与周末
#    两种完全不同的出行模式，帮助模型快速捕捉宏观差异。
# 5. is_rush_hour：是否高峰时段。高峰时段是通勤需求集中释放的时间段，
#    需求量显著高于平峰时段，是对小时维度的业务规则补充，强化高峰效应特征。
# --------------------------
feature_cols = ['PULocationID', 'pickup_hour', 'pickup_weekday', 'is_weekend', 'is_rush_hour']
X = demand_data[feature_cols].copy()
y = demand_data['demand'].values

# 将分类特征转为类别类型，做独热编码
categorical_cols = ['PULocationID', 'pickup_hour', 'pickup_weekday']
for col in categorical_cols:
    X[col] = X[col].astype('category')
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

# --------------------------
# 按8:2划分训练集与测试集
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_SEED
)

# 特征标准化：神经网络对特征尺度敏感，标准化后训练更稳定；随机森林不受尺度影响
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"数据集构造完成，总样本数：{len(demand_data)}")
print(f"训练集样本数：{len(X_train)}，测试集样本数：{len(X_test)}")
print(f"输入特征维度：{X_encoded.shape[1]}")

# --------------------------
# 模型1：随机森林回归（基线模型）
# --------------------------
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
# 随机森林对特征尺度不敏感，直接使用原始编码特征训练
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 计算评估指标
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

# --------------------------
# 模型2：PyTorch 全连接神经网络（MLP）
# --------------------------
# 转换为PyTorch张量格式
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# 定义神经网络结构
class DemandPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DemandPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout抑制过拟合
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # 回归任务，输出单个预测值
        )

    def forward(self, x):
        return self.layers(x)


# 初始化模型、损失函数、优化器
input_dim = X_train_scaled.shape[1]
nn_model = DemandPredictor(input_dim)
criterion = nn.MSELoss()  # 回归任务使用均方误差损失
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# 训练参数
epochs = 150
train_losses = []

# 训练循环
nn_model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = nn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    if (epoch + 1) % 30 == 0:
        print(f"神经网络训练 Epoch [{epoch + 1}/{epochs}], 训练Loss: {loss.item():.4f}")

# 绘制并保存训练Loss曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='训练损失', color='#1f77b4', linewidth=2)
plt.title('神经网络训练损失变化曲线', fontsize=14, pad=15)
plt.xlabel('训练轮次 Epoch', fontsize=12)
plt.ylabel('MSE 损失值', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/m3_neural_network_loss.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ 神经网络训练损失曲线已保存：outputs/m3_neural_network_loss.png")

# 测试集评估
nn_model.eval()
with torch.no_grad():
    nn_pred = nn_model(X_test_tensor).numpy().flatten()

nn_mae = mean_absolute_error(y_test, nn_pred)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))

# --------------------------
# 保存模型评估指标
# --------------------------
metrics_df = pd.DataFrame({
    '模型名称': ['随机森林', '全连接神经网络(MLP)'],
    'MAE_平均绝对误差': [round(rf_mae, 4), round(nn_mae, 4)],
    'RMSE_均方根误差': [round(rf_rmse, 4), round(nn_rmse, 4)]
})
metrics_df.to_csv('outputs/m3_model_metrics.csv', index=False, encoding='utf-8-sig')
print("✅ 模型评估指标已保存：outputs/m3_model_metrics.csv")
print("\n===== 测试集评估结果 =====")
print(metrics_df.to_string(index=False))

# --------------------------
# 两种模型方法优劣分析
# 1. 随机森林
#    优势：
#    - 对特征尺度不敏感，无需复杂预处理，类别特征编码后可直接使用，工程成本低
#    - 可解释性强，支持输出特征重要性，能直观判断区域、时段等因素的影响权重
#    - 训练速度快，调参成本低，在表格数据、中小数据集上表现稳定，不易过拟合
#    - 对异常值和噪声的鲁棒性较好，适配出行需求这类存在自然波动的业务数据
#    劣势：
#    - 泛化能力有限，对训练集中未出现的区域-时段组合，预测效果会明显下降
#    - 模型容量有限，难以捕捉特征间高阶、复杂的非线性交互关系
#    - 数据量大幅增长时，性能提升存在明显瓶颈
#
# 2. 全连接神经网络
#    优势：
#    - 模型容量大，能够拟合特征间复杂的非线性交互关系，数据量充足时上限更高
#    - 扩展性强，后续可灵活加入嵌入层、时序卷积、注意力等结构，适配更丰富的特征
#    - 随着数据量增加，性能可以持续提升，适合长期迭代的业务场景
#    劣势：
#    - 对数据预处理要求高，特征需要标准化，对类别特征的编码方式敏感
#    - 可解释性差，属于黑盒模型，难以量化每个特征对预测结果的具体影响
#    - 小数据集上容易过拟合，需要Dropout、正则化等手段约束模型
#    - 训练调参成本高，需要调整网络深度、宽度、学习率、训练轮次等多个超参数
#
# 本任务场景结论：
# 区域时段出行需求预测属于典型的表格数据回归任务，特征以类别型为主，
# 在当前单月数据量下，随机森林通常表现更稳定且解释性更强，是更优的工程选择；
# 当积累多月/多年数据、加入天气、节假日等更多外部特征后，神经网络的优势会逐步显现。
# --------------------------

import re

# ==================================================
# 命令行智能问答系统
# 支持6类问题：时段订单查询、区域热度排名、车费统计、需求预测、模型效果、数据质量
# ==================================================

# 前置准备：缓存训练集特征列名，用于预测时特征维度对齐
train_feature_columns = X_encoded.columns.tolist()


# --------------------------
# 工具函数：数字提取与中文时间解析
# --------------------------
def extract_numbers(text):
    """从文本中提取所有整数，返回列表"""
    return [int(num) for num in re.findall(r'\d+', text)]


def chinese_hour_to_num(text):
    """中文模糊时间表述转小时数，匹配不到返回None"""
    time_keywords = {
        '凌晨': 2, '早上': 8, '上午': 10, '中午': 12,
        '下午': 15, '傍晚': 18, '晚上': 20, '深夜': 23,
        '早高峰': 8, '晚高峰': 18
    }
    for kw, hour in time_keywords.items():
        if kw in text:
            return hour
    return None


# --------------------------
# 查询函数1：时段订单量统计
# --------------------------
def query_hourly_demand(params):
    """
    查询指定时段的出行订单量
    入参：hour(小时), is_weekend(是否周末, None=全部日期)
    返回：(数字结论, 文本解释, 相关文件路径列表)
    """
    hour = params.get('hour')
    is_weekend = params.get('is_weekend')
    df = df_clean.copy()
    desc_tags = []

    # 日期类型筛选
    if is_weekend is not None:
        df = df[df['is_weekend'] == is_weekend]
        desc_tags.append("周末" if is_weekend else "工作日")

    # 小时筛选
    if hour is not None:
        df = df[df['pickup_hour'] == hour]
        desc_tags.append(f"{hour}点时段")
        total_orders = len(df)
        # 计算日均订单量
        day_count = df['pickup_date'].nunique()
        avg_daily = round(total_orders / day_count, 1) if day_count > 0 else 0

        is_rush = (7 <= hour < 9 or 17 <= hour < 19) and is_weekend == 0
        explain = (
            f"{' '.join(desc_tags)}总订单量为{total_orders}单，平均每日约{avg_daily}单。"
            f"该时段{'属于通勤高峰，出行需求显著高于平峰' if is_rush else '处于平峰阶段，需求相对平稳'}。"
        )
        result_num = avg_daily
    else:
        # 未指定小时，返回全天平均每小时订单量
        hourly_avg = round(df.groupby('pickup_hour').size().mean(), 1)
        result_num = hourly_avg
        explain = f"统计范围内平均每小时订单量约{hourly_avg}单，全天需求呈现早晚双峰分布，8点、18点为全天需求峰值。"

    files = ["outputs/m2_1_time_demand.png"]
    return result_num, explain, files


# --------------------------
# 查询函数2：区域热度排名
# --------------------------
def query_top_zones(params):
    """
    查询上下车热门区域TOP排名
    入参：top_n(排名数量), zone_type(上车/下车)
    """
    top_n = params.get('top_n', 10)
    zone_type = params.get('zone_type', '上车')
    col = 'PULocationID' if zone_type == '上车' else 'DOLocationID'

    top_series = df_clean[col].value_counts().head(top_n)
    top_list = [f"第{i + 1}名：区域{zone_id}（{count}单）" for i, (zone_id, count) in enumerate(top_series.items())]

    result_num = top_series.iloc[0]
    explain = (
            f"{zone_type}订单量TOP{top_n}区域如下：\n" + "\n".join(top_list) +
            f"\n\n排名第一的区域{top_series.index[0]}订单量最高，为城市核心出行热点区域。"
    )
    files = ["outputs/m2_2_top10_zone.png", "outputs/m2_2_zone_heatmap.png"]
    return result_num, explain, files


# --------------------------
# 查询函数3：车费统计分析
# --------------------------
def query_fare_stats(params):
    """
    查询不同场景下的车费统计
    入参：scope(整体/高峰/非高峰/工作日/周末)
    """
    scope = params.get('scope', '整体')
    df = df_clean.copy()

    scope_map = {
        '高峰': 'is_rush_hour == 1',
        '非高峰': 'is_rush_hour == 0',
        '工作日': 'is_weekend == 0',
        '周末': 'is_weekend == 1'
    }
    if scope in scope_map:
        df = df.query(scope_map[scope])

    avg_fare = round(df['total_amount'].mean(), 2)
    median_fare = round(df['total_amount'].median(), 2)

    result_num = avg_fare
    explain = (
        f"{scope}行程平均总车费为{avg_fare}美元，中位数为{median_fare}美元。"
        f"车费核心由里程费构成，高峰时段因拥堵时长增加，单位里程成本略有上升。"
    )
    files = ["outputs/m2_3_fare_factors.png"]
    return result_num, explain, files


# --------------------------
# 查询函数4：区域时段需求量预测
# --------------------------
def predict_zone_demand(params):
    """
    基于随机森林模型预测指定区域时段的出行需求量
    入参：zone_id(区域ID), hour(小时), weekday(星期,0=周一)
    """
    zone_id = params.get('zone_id')
    hour = params.get('hour')
    weekday = params.get('weekday', 0)  # 默认周一

    if zone_id is None or hour is None:
        return None, "请提供具体的区域ID和小时数，例如：预测区域100周一早上8点的需求量", []

    # 计算衍生特征
    is_weekend = 1 if weekday >= 5 else 0
    is_rush = 1 if (not is_weekend) and ((7 <= hour < 9) or (17 <= hour < 19)) else 0

    # 构造输入样本
    input_row = pd.DataFrame([{
        'PULocationID': zone_id,
        'pickup_hour': hour,
        'pickup_weekday': weekday,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush
    }])

    # 分类特征独热编码，与训练集对齐
    cat_cols = ['PULocationID', 'pickup_hour', 'pickup_weekday']
    for col in cat_cols:
        input_row[col] = input_row[col].astype('category')
    input_encoded = pd.get_dummies(input_row, columns=cat_cols, drop_first=False)
    input_encoded = input_encoded.reindex(columns=train_feature_columns, fill_value=0)

    # 模型预测
    pred_value = round(rf_model.predict(input_encoded)[0], 1)
    weekday_name = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][weekday]

    result_num = pred_value
    explain = (
        f"预测{weekday_name}{hour}点，区域{zone_id}的出行需求量约为{pred_value}单。"
        f"{'该时段为通勤高峰，需求处于较高水平' if is_rush else '该时段为平峰时段，需求相对平稳'}。"
    )
    files = ["outputs/m3_model_metrics.csv", "outputs/m3_neural_network_loss.png"]
    return result_num, explain, files


# --------------------------
# 查询函数5：模型效果对比
# --------------------------
def query_model_performance(params):
    """查询随机森林与神经网络的预测效果对比"""
    rf_mae = metrics_df.iloc[0]['MAE_平均绝对误差']
    rf_rmse = metrics_df.iloc[0]['RMSE_均方根误差']
    nn_mae = metrics_df.iloc[1]['MAE_平均绝对误差']
    nn_rmse = metrics_df.iloc[1]['RMSE_均方根误差']

    better_model = "随机森林" if rf_mae < nn_mae else "全连接神经网络"
    result_num = round(min(rf_mae, nn_mae), 4)

    explain = (
        f"【随机森林】MAE = {rf_mae}，RMSE = {rf_rmse}\n"
        f"【全连接神经网络】MAE = {nn_mae}，RMSE = {nn_rmse}\n\n"
        f"当前数据集下{better_model}预测精度更优。随机森林在表格数据上表现稳定、可解释性强；"
        f"神经网络在数据量充足、特征丰富时性能上限更高。"
    )
    files = ["outputs/m3_model_metrics.csv", "outputs/m3_neural_network_loss.png"]
    return result_num, explain, files


# --------------------------
# 查询函数6：数据质量概况
# --------------------------
def query_data_quality(params):
    """查询数据集整体质量情况"""
    total_raw = len(df_raw)
    total_clean = len(df_clean)
    drop_count = total_raw - total_clean

    # 筛选数值型缺失率，取最高字段
    valid_missing = quality_report[quality_report['缺失率(%)'] != '-'].copy()
    valid_missing['缺失率(%)'] = valid_missing['缺失率(%)'].astype(float)
    max_missing_row = valid_missing.loc[valid_missing['缺失率(%)'].idxmax()]

    result_num = drop_count
    explain = (
        f"原始数据共{total_raw}条，清洗后剩余{total_clean}条，累计剔除异常/无效数据{drop_count}条。\n"
        f"缺失率最高的字段是「{max_missing_row['字段名称']}」，缺失率为{max_missing_row['缺失率(%)']}%。\n"
        f"整体数据质量良好，核心字段完整度高，已完成负值校验、逻辑异常、枚举合法性、极端值多维度清洗。"
    )
    files = ["outputs/data_quality_report.csv"]
    return result_num, explain, files


# --------------------------
# 意图识别与参数提取
# --------------------------
def recognize_intent(user_text):
    """
    基于关键词规则匹配意图，提取对应参数
    返回：(意图标识, 参数字典)
    """
    text = user_text.strip()
    params = {}

    # 1. 需求量预测（优先级最高，避免关键词冲突）
    predict_kws = ['预测', '预计', '估算', '预估', '需求量']
    if any(kw in text for kw in predict_kws):
        nums = extract_numbers(text)
        if len(nums) >= 2:
            params['zone_id'], params['hour'] = nums[0], nums[1]
        elif len(nums) == 1:
            params['hour'] = nums[0]

        # 识别星期
        weekday_map = {'周一': 0, '周二': 1, '周三': 2, '周四': 3, '周五': 4, '周六': 5, '周日': 6}
        for wd, val in weekday_map.items():
            if wd in text:
                params['weekday'] = val
                break
        if '周末' in text:
            params['weekday'] = 5
        return 'predict_demand', params

    # 2. 区域热度排名
    rank_kws = ['排名', 'top', '热门', '最多', '最热', '哪些区域', '前几']
    if any(kw in text for kw in rank_kws):
        nums = extract_numbers(text)
        params['top_n'] = nums[0] if nums else 10
        params['zone_type'] = '下车' if ('下车' in text or '目的地' in text) else '上车'
        return 'top_zones', params

    # 3. 车费统计查询
    fare_kws = ['车费', '费用', '价格', '平均多少钱', '花费']
    if any(kw in text for kw in fare_kws):
        if '高峰' in text:
            params['scope'] = '高峰'
        elif '非高峰' in text:
            params['scope'] = '非高峰'
        elif '工作日' in text:
            params['scope'] = '工作日'
        elif '周末' in text:
            params['scope'] = '周末'
        else:
            params['scope'] = '整体'
        return 'fare_stats', params

    # 4. 模型效果查询
    model_kws = ['模型', '误差', '准确率', '哪个好', '效果', '精度']
    if any(kw in text for kw in model_kws):
        return 'model_performance', params

    # 5. 数据质量查询
    quality_kws = ['数据质量', '缺失', '异常', '多少条数据', '清洗', '数据量']
    if any(kw in text for kw in quality_kws):
        return 'data_quality', params

    # 6. 时段订单量查询（兜底匹配）
    hour_kws = ['点', '小时', '时段', '订单量', '多少单', '几点']
    if any(kw in text for kw in hour_kws):
        nums = extract_numbers(text)
        ch_hour = chinese_hour_to_num(text)
        if nums:
            params['hour'] = nums[0]
        elif ch_hour:
            params['hour'] = ch_hour

        if '周末' in text or '周六' in text or '周日' in text:
            params['is_weekend'] = 1
        elif '工作日' in text or '周一' in text or '周二' in text or '周三' in text or '周四' in text or '周五' in text:
            params['is_weekend'] = 0
        return 'hourly_demand', params

    # 未识别意图
    return 'unknown', params


# --------------------------
# 主问答循环
# --------------------------
def run_chatbot():
    print("\n" + "=" * 55)
    print("🚕 纽约黄色出租车数据智能问答系统")
    print("=" * 55)
    print("支持以下6类问题，输入 exit 退出：")
    print("  1. 时段订单：工作日早上8点有多少订单？")
    print("  2. 区域排名：上车最多的前10个区域")
    print("  3. 车费查询：高峰时段平均车费多少？")
    print("  4. 需求预测：预测区域100周一早8点的需求量")
    print("  5. 模型对比：两个模型哪个预测更准？")
    print("  6. 数据质量：数据有多少缺失值？")
    print("=" * 55)

    # 意图与处理函数映射
    handler_map = {
        'hourly_demand': query_hourly_demand,
        'top_zones': query_top_zones,
        'fare_stats': query_fare_stats,
        'predict_demand': predict_zone_demand,
        'model_performance': query_model_performance,
        'data_quality': query_data_quality
    }

    while True:
        user_input = input("\n💬 请输入问题：").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', '退出', 'q', 'quit']:
            print("👋 感谢使用，再见！")
            break

        # 意图识别
        intent, params = recognize_intent(user_input)
        if intent == 'unknown':
            print("❌ 暂时无法理解该问题，您可以尝试询问时段、区域、车费、预测、模型、数据质量相关内容。")
            continue

        # 调用对应处理函数
        handler = handler_map[intent]
        num_result, explain, file_list = handler(params)

        if num_result is None:
            print(f"⚠️  {explain}")
            continue

        # 格式化输出结果
        print("\n" + "-" * 50)
        print(f"📊 数字结论：{num_result}")
        print(f"💡 解释说明：{explain}")
        if file_list:
            print("📁 相关文件：")
            for f in file_list:
                print(f"     {f}")
        print("-" * 50)


# 启动问答系统
run_chatbot()