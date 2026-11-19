# ==============================================
# 游戏用户行为数据集生成脚本
# 输出：game_user_data.csv，共10万条用户数据
# ==============================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# --------------------------
# 1. 基础参数设置
# --------------------------
np.random.seed(42)  # 固定随机种子，保证每次运行结果一致
random.seed(42)
total_users = 100000  # 用户总数
date_end = datetime(2026, 11, 20)  # 数据截止日期
date_start = date_end - timedelta(days=90)  # 数据起始日期（90天前）

print(f"开始生成 {total_users} 条用户数据...")
print(f"数据时间范围：{date_start.date()} 至 {date_end.date()}")

# --------------------------
# 2. 生成用户基础信息
# --------------------------
# 生成用户ID
user_ids = [f"user_{i:06d}" for i in range(1, total_users + 1)]

# 生成注册日期：模拟新用户逐渐衰减的注册趋势，前期注册多，后期少
register_weights = np.array([(90 - i) for i in range(90)], dtype=float)
register_probs = register_weights / register_weights.sum()  # 归一化，强制概率和为1
register_days = np.random.choice(range(1, 91), size=total_users, p=register_probs)
register_dates = [date_end - timedelta(days=int(d)) for d in register_days]

# 设备类型：安卓70%，iOS30%
device_types = np.random.choice(["Android", "iOS"], size=total_users, p=[0.7, 0.3])

# --------------------------
# 3. 生成用户活跃行为数据
# --------------------------
# 计算每个用户的注册天数（距截止日期）
days_since_register = [(date_end - d).days for d in register_dates]

# 累计登录天数：基于注册天数，模拟留存自然衰减
# 注册天数越多，累计登录天数上限越高，但留存率随时间下降
total_login_days = []
for days in days_since_register:
    # 留存率随时间衰减：第1天约60%，第7天约30%，第30天约10%
    retention_rate = 0.6 * np.exp(-0.05 * days) + 0.05
    login_days = np.random.binomial(days, min(retention_rate, 0.8))
    total_login_days.append(max(1, login_days))  # 至少登录1天（注册当天）

# 最后登录日期 = 注册日期 + 随机分布的最后活跃偏移
last_login_dates = []
for i in range(total_users):
    reg_date = register_dates[i]
    login_d = total_login_days[i]
    days_reg = days_since_register[i]
    
    # 计算最大可偏移天数，边界保护：不能小于0
    max_offset = max(0, days_reg - login_d)
    
    # 大部分用户近期活跃过，流失用户则较早停止登录
    if login_d / max(days_reg, 1) > 0.3:  # 活跃用户
        upper = min(max_offset, 7)
        last_offset = random.randint(0, upper)
    else:  # 流失用户
        lower = min(days_reg // 2, max_offset)
        last_offset = random.randint(lower, max_offset)
    
    last_date = reg_date + timedelta(days=days_reg - last_offset)
    last_login_dates.append(last_date.date())

# 日均游戏时长：正态分布，付费用户时长更高
avg_daily_minutes = np.random.normal(loc=45, scale=20, size=total_users)
avg_daily_minutes = np.clip(avg_daily_minutes, 5, 180)  # 限制在5-180分钟范围内

# --------------------------
# 4. 生成游戏进度数据
# --------------------------
# 关卡通关数：与累计登录天数正相关，大部分用户卡在中期关卡（20-40关）
base_level = np.array(total_login_days) * 0.8  # 基础关卡进度
level_noise = np.random.normal(0, 5, total_users)  # 随机波动
level_passed = np.round(base_level + level_noise).astype(int)
level_passed = np.clip(level_passed, 1, 100)  # 最多100关

# 用户等级：与关卡数正相关，每5关升1级左右
user_level = np.round(level_passed / 5 + 1).astype(int)
user_level = np.clip(user_level, 1, 30)

# 新手教程完成率：92%的用户完成
tutorial_finished = np.random.choice([0, 1], size=total_users, p=[0.08, 0.92])

# --------------------------
# 5. 生成付费行为数据
# --------------------------
# 付费率：整体约7.5%，活跃用户付费率更高
pay_prob = 0.03 + 0.1 * (np.array(total_login_days) / max(total_login_days))
is_paid = np.array([np.random.binomial(1, min(p, 0.2)) for p in pay_prob])

# 累计充值金额：付费用户呈现长尾分布（大部分小额，少数大额）
total_pay_amount = np.zeros(total_users)
first_pay_dates = [None] * total_users

for i in range(total_users):
    if is_paid[i] == 1:
        # 帕累托分布模拟付费长尾：80%付费来自20%用户
        amount = np.random.pareto(2) * 30 + 6  # 最低6元首充
        total_pay_amount[i] = round(amount, 2)
        
        # 首充时间：一般在注册后1-7天内
        days_after_reg = random.randint(1, min(7, days_since_register[i]))
        first_pay_dates[i] = (register_dates[i] + timedelta(days=days_after_reg)).date()
        
        # 付费用户游戏时长和关卡进度略高
        avg_daily_minutes[i] *= 1.3
        level_passed[i] = int(level_passed[i] * 1.2)

# --------------------------
# 6. 加入缺失值和异常值（约5%）
# --------------------------
# 转为DataFrame
df = pd.DataFrame({
    "user_id": user_ids,
    "register_date": [d.date() for d in register_dates],
    "last_login_date": last_login_dates,
    "total_login_days": total_login_days,
    "avg_daily_minutes": np.round(avg_daily_minutes, 1),
    "level_passed": level_passed,
    "is_paid": is_paid,
    "total_pay_amount": total_pay_amount,
    "first_pay_date": first_pay_dates,
    "tutorial_finished": tutorial_finished,
    "user_level": user_level,
    "device_type": device_types
})

# 加入缺失值：各字段随机设置约3-5%的空值
missing_rate = 0.04
for col in ["avg_daily_minutes", "total_pay_amount", "tutorial_finished", "device_type"]:
    mask = np.random.choice([True, False], size=total_users, p=[missing_rate, 1-missing_rate])
    df.loc[mask, col] = np.nan

# 加入异常值：约1%的异常数据
# 异常高的游戏时长
outlier_idx = np.random.choice(total_users, size=int(total_users*0.005), replace=False)
df.loc[outlier_idx, "avg_daily_minutes"] = np.random.randint(500, 1000, size=len(outlier_idx))

# 异常高的充值金额
outlier_idx2 = np.random.choice(total_users, size=int(total_users*0.005), replace=False)
df.loc[outlier_idx2, "total_pay_amount"] = np.random.randint(10000, 50000, size=len(outlier_idx2))

# --------------------------
# 7. 保存为CSV文件
# --------------------------
df.to_csv("game_user_data.csv", index=False, encoding="utf-8-sig")

# --------------------------
# 8. 输出数据概览
# --------------------------
print("=" * 60)
print(f"✅ 数据集生成完成！已保存为 game_user_data.csv")
print("=" * 60)
print(f"总用户数：{len(df)}")
print(f"付费用户数：{df['is_paid'].sum()}")
print(f"付费率：{df['is_paid'].mean():.2%}")
print(f"缺失值总数：{df.isnull().sum().sum()}")
print(f"数据时间跨度：{df['register_date'].min()} 至 {df['last_login_date'].max()}")
print(f"平均累计登录天数：{df['total_login_days'].mean():.1f} 天")
print(f"平均关卡通关数：{df['level_passed'].mean():.1f} 关")
print(f"平均累计充值（付费用户）：{df[df['is_paid']==1]['total_pay_amount'].mean():.2f} 元")
print("=" * 60)
