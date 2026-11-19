# ==============================================
# 阶段3：数据清洗 + SQLite核心指标计算
# 输出：cleaned_game_data.csv（清洗后数据）、game_analysis.db（数据库文件）
# ==============================================
import pandas as pd
import numpy as np
import sqlite3

# ==============================================
# 第一部分：数据清洗
# ==============================================
print("=" * 60)
print("🧹 开始执行数据清洗")
print("=" * 60)

# 1. 读取原始数据集
df_raw = pd.read_csv("game_user_data.csv", encoding="utf-8-sig")
print(f"\n📊 清洗前数据概览")
print(f"原始数据总行数：{len(df_raw)}")
print(f"原始数据总列数：{df_raw.shape[1]}")
print("\n各字段缺失值数量统计：")
print(df_raw.isnull().sum())

# 复制一份数据用于清洗，不修改原始数据
df = df_raw.copy()

# --------------------------
# 2. 缺失值处理（分字段差异化策略，附业务原因）
# --------------------------
# 2.1 累计充值金额
# 业务逻辑：未付费用户本身充值为0，缺失直接补0；付费用户缺失用中位数填充，避免拉低付费水平
df.loc[df['is_paid'] == 0, 'total_pay_amount'] = df.loc[df['is_paid'] == 0, 'total_pay_amount'].fillna(0)
paid_median = df[df['is_paid'] == 1]['total_pay_amount'].median()
df.loc[df['is_paid'] == 1, 'total_pay_amount'] = df.loc[df['is_paid'] == 1, 'total_pay_amount'].fillna(paid_median)

# 2.2 日均游戏时长
# 业务逻辑：时长缺失多为客户端埋点上报失败，中位数填充对整体数据分布影响最小
duration_median = df['avg_daily_minutes'].median()
df['avg_daily_minutes'] = df['avg_daily_minutes'].fillna(duration_median)

# 2.3 新手教程完成状态
# 业务逻辑：绝大多数用户会完成教程，用众数填充符合真实用户行为分布
tutorial_mode = df['tutorial_finished'].mode()[0]
df['tutorial_finished'] = df['tutorial_finished'].fillna(tutorial_mode)

# 2.4 设备类型
# 业务逻辑：设备信息不影响核心分析，缺失单独归为未知类，保留用户样本完整性
df['device_type'] = df['device_type'].fillna('Unknown')

# 2.5 核心标识字段（用户ID、注册/最后登录日期）
# 业务逻辑：这些是用户分析的基础，缺失无法追溯用户生命周期，直接删除
df = df.dropna(subset=['user_id', 'register_date', 'last_login_date'])

# --------------------------
# 3. 异常值识别与处理
# --------------------------
print("\n" + "=" * 60)
print("⚠️  异常值识别与处理结果")
print("=" * 60)

# 3.1 日均游戏时长异常：阈值设为300分钟（5小时）
# 业务逻辑：普通用户日均游戏时长极少超过5小时，极端值多为后台挂机或埋点错误
duration_outlier_count = len(df[df['avg_daily_minutes'] > 300])
print(f"日均时长超300分钟的异常用户数：{duration_outlier_count}")
# 处理方式：盖帽修正，超过300的统一设为300，保留用户样本
df['avg_daily_minutes'] = np.clip(df['avg_daily_minutes'], 0, 300)

# 3.2 累计充值金额异常：阈值设为2000元
# 业务逻辑：模拟的是中度手游，普通付费用户累计充值极少超过2000元
pay_outlier_count = len(df[(df['is_paid'] == 1) & (df['total_pay_amount'] > 2000)])
print(f"累计充值超2000元的异常用户数：{pay_outlier_count}")
# 处理方式：盖帽修正
df.loc[df['total_pay_amount'] > 2000, 'total_pay_amount'] = 2000

# --------------------------
# 4. 数据格式统一
# --------------------------
# 4.1 日期字段转为标准时间格式，方便后续时间计算
df['register_date'] = pd.to_datetime(df['register_date'])
df['last_login_date'] = pd.to_datetime(df['last_login_date'])
df['first_pay_date'] = pd.to_datetime(df['first_pay_date'])

# 4.2 数值类型标准化，保证数据类型一致
df['total_login_days'] = df['total_login_days'].astype(int)
df['level_passed'] = df['level_passed'].astype(int)
df['is_paid'] = df['is_paid'].astype(int)
df['tutorial_finished'] = df['tutorial_finished'].astype(int)
df['user_level'] = df['user_level'].astype(int)
df['total_pay_amount'] = df['total_pay_amount'].round(2)
df['avg_daily_minutes'] = df['avg_daily_minutes'].round(1)

# --------------------------
# 5. 清洗结果输出与保存
# --------------------------
print("\n" + "=" * 60)
print("✅ 清洗后数据概览")
print("=" * 60)
print(f"清洗后总行数：{len(df)}")
print(f"数据清洗损失率：{(1 - len(df)/len(df_raw)):.2%}")
print("\n清洗后缺失值统计（应全为0）：")
print(df.isnull().sum())

# 保存清洗后的数据
df.to_csv("cleaned_game_data.csv", index=False, encoding="utf-8-sig")
print("\n✅ 清洗后数据已保存为 cleaned_game_data.csv")

# ==============================================
# 第二部分：SQLite 核心业务指标计算
# ==============================================
print("\n" + "=" * 60)
print("🗄️  SQLite数据库指标计算")
print("=" * 60)

# 1. 连接数据库（自动创建db文件，无需额外安装数据库软件）
conn = sqlite3.connect('game_analysis.db')
cursor = conn.cursor()
print("已连接数据库：game_analysis.db")

# 2. 建表并导入数据（重复运行先删旧表）
cursor.execute('DROP TABLE IF EXISTS user_behavior')
create_table_sql = '''
CREATE TABLE user_behavior (
    user_id TEXT PRIMARY KEY,
    register_date TEXT,
    last_login_date TEXT,
    total_login_days INTEGER,
    avg_daily_minutes REAL,
    level_passed INTEGER,
    is_paid INTEGER,
    total_pay_amount REAL,
    first_pay_date TEXT,
    tutorial_finished INTEGER,
    user_level INTEGER,
    device_type TEXT
)
'''
cursor.execute(create_table_sql)
conn.commit()

# 将清洗后的DataFrame导入数据库表
df.to_sql('user_behavior', conn, if_exists='append', index=False)
print("已将清洗后数据导入 user_behavior 表")

# 分析基准日期（与数据集截止日期一致）
analysis_date = '2026-11-20'

# --------------------------
# 指标1：DAU（日活跃用户数）
# 定义：截止当日有登录行为的用户总数
# --------------------------
print("\n📌 指标1：DAU（日活跃用户数）")
dau_sql = f'''
SELECT COUNT(DISTINCT user_id) AS DAU
FROM user_behavior
WHERE last_login_date = '{analysis_date}'
'''
dau_result = pd.read_sql(dau_sql, conn)
print(f"截止 {analysis_date} 的DAU：{dau_result.iloc[0, 0]} 人")

# --------------------------
# 指标2：整体付费率
# 定义：付费用户数 / 总用户数
# --------------------------
print("\n📌 指标2：整体付费率")
pay_rate_sql = '''
SELECT 
    COUNT(CASE WHEN is_paid = 1 THEN user_id END) AS paid_users,
    COUNT(user_id) AS total_users,
    ROUND(COUNT(CASE WHEN is_paid = 1 THEN user_id END) * 100.0 / COUNT(user_id), 2) AS pay_rate
FROM user_behavior
'''
pay_rate_result = pd.read_sql(pay_rate_sql, conn)
print(f"付费用户数：{pay_rate_result.iloc[0, 0]} 人")
print(f"总用户数：{pay_rate_result.iloc[0, 1]} 人")
print(f"整体付费率：{pay_rate_result.iloc[0, 2]}%")

# --------------------------
# 指标3：ARPU & ARPPU
# ARPU：每用户平均收入（总收入/总用户）
# ARPPU：每付费用户平均收入（总收入/付费用户）
# --------------------------
print("\n📌 指标3：ARPU & ARPPU")
arpu_sql = '''
SELECT 
    ROUND(SUM(total_pay_amount) / COUNT(user_id), 2) AS ARPU,
    ROUND(SUM(total_pay_amount) / COUNT(CASE WHEN is_paid = 1 THEN user_id END), 2) AS ARPPU
FROM user_behavior
'''
arpu_result = pd.read_sql(arpu_sql, conn)
print(f"ARPU（全用户人均收入）：{arpu_result.iloc[0, 0]} 元")
print(f"ARPPU（付费用户人均收入）：{arpu_result.iloc[0, 1]} 元")

# --------------------------
# 指标4：次日留存率
# 定义：注册满1天的用户中，注册次日仍登录的用户占比
# --------------------------
print("\n📌 指标4：次日留存率")
retention_1d_sql = '''
SELECT 
    COUNT(CASE WHEN julianday(last_login_date) - julianday(register_date) >= 1 THEN user_id END) AS retain_1d,
    COUNT(CASE WHEN julianday('2026-11-20') - julianday(register_date) >= 1 THEN user_id END) AS eligible_users,
    ROUND(
        COUNT(CASE WHEN julianday(last_login_date) - julianday(register_date) >= 1 THEN user_id END) * 100.0 
        / COUNT(CASE WHEN julianday('2026-11-20') - julianday(register_date) >= 1 THEN user_id END), 
    2) AS retention_1d_rate
FROM user_behavior
'''
retention_1d_result = pd.read_sql(retention_1d_sql, conn)
print(f"注册满1天的用户数：{retention_1d_result.iloc[0, 1]} 人")
print(f"次日留存用户数：{retention_1d_result.iloc[0, 0]} 人")
print(f"次日留存率：{retention_1d_result.iloc[0, 2]}%")

# --------------------------
# 指标5：7日留存率
# 定义：注册满7天的用户中，第7天仍登录的用户占比
# --------------------------
print("\n📌 指标5：7日留存率")
retention_7d_sql = '''
SELECT 
    COUNT(CASE WHEN julianday(last_login_date) - julianday(register_date) >= 7 THEN user_id END) AS retain_7d,
    COUNT(CASE WHEN julianday('2026-11-20') - julianday(register_date) >= 7 THEN user_id END) AS eligible_users,
    ROUND(
        COUNT(CASE WHEN julianday(last_login_date) - julianday(register_date) >= 7 THEN user_id END) * 100.0 
        / COUNT(CASE WHEN julianday('2026-11-20') - julianday(register_date) >= 7 THEN user_id END), 
    2) AS retention_7d_rate
FROM user_behavior
'''
retention_7d_result = pd.read_sql(retention_7d_sql, conn)
print(f"注册满7天的用户数：{retention_7d_result.iloc[0, 1]} 人")
print(f"7日留存用户数：{retention_7d_result.iloc[0, 0]} 人")
print(f"7日留存率：{retention_7d_result.iloc[0, 2]}%")

# 关闭数据库连接
conn.close()
print("\n" + "=" * 60)
print("✅ 所有SQL指标计算完成，数据库连接已关闭")
print("=" * 60)
