# ==============================================
# 阶段5：用户流失预测模型（随机森林）
# 输出：模型评估指标、特征重要性排序、流失干预建议
# 技术栈：scikit-learn
# ==============================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# --------------------------
# 0. 数据读取与基础准备
# --------------------------
df = pd.read_csv("cleaned_game_data.csv", encoding="utf-8-sig")
# 统一日期格式
df['register_date'] = pd.to_datetime(df['register_date'])
df['last_login_date'] = pd.to_datetime(df['last_login_date'])
analysis_date = pd.to_datetime('2026-11-20')

print("=" * 60)
print("🤖 开始构建用户流失预测模型")
print("=" * 60)

# ==============================================
# 1. 定义流失标签（游戏行业通用标准）
# 规则：连续7天及以上未登录，定义为流失用户
# 标签：1=流失，0=未流失
# ==============================================
# 计算用户距离最后登录的天数
df['days_since_last_login'] = (analysis_date - df['last_login_date']).dt.days

# 生成流失标签
df['is_churn'] = (df['days_since_last_login'] >= 7).astype(int)

churn_count = df['is_churn'].sum()
total_count = len(df)
print(f"\n📌 流失标签统计：")
print(f"  流失用户数：{churn_count} 人")
print(f"  未流失用户数：{total_count - churn_count} 人")
print(f"  整体流失率：{churn_count / total_count:.2%}")
print(f"  （样本存在不平衡，后续将通过样本权重处理）")

# ==============================================
# 2. 特征工程：衍生10+用户行为特征
# 通俗解释：把原始数据加工成模型能识别的、和流失相关的指标
# ==============================================
# 计算注册时长（天）
df['register_days'] = (analysis_date - df['register_date']).dt.days

# 衍生特征列表
features = pd.DataFrame()
features['total_login_days'] = df['total_login_days']          # 累计登录天数
features['avg_daily_minutes'] = df['avg_daily_minutes']        # 日均游戏时长
features['level_passed'] = df['level_passed']                  # 通关关卡数
features['user_level'] = df['user_level']                      # 用户等级
features['tutorial_finished'] = df['tutorial_finished']        # 是否完成新手教程
features['is_paid'] = df['is_paid']                            # 是否付费
features['total_pay_amount'] = df['total_pay_amount']          # 累计充值金额
features['register_days'] = df['register_days']                # 注册时长
features['active_rate'] = df['total_login_days'] / df['register_days']  # 用户活跃率
features['avg_level_per_day'] = df['level_passed'] / df['total_login_days']  # 日均通关数
features['pay_intensity'] = df['total_pay_amount'] / (df['total_login_days'] + 1)  # 日均付费强度

# 类别特征编码：设备类型转为数值
device_dummies = pd.get_dummies(df['device_type'], prefix='device', drop_first=True)
features = pd.concat([features, device_dummies], axis=1)

# 填充可能的无穷值/空值（防止除零异常）
features = features.replace([np.inf, -np.inf], 0)
features = features.fillna(0)

# 标签
labels = df['is_churn']

print(f"\n✅ 特征工程完成，共生成 {features.shape[1]} 个特征")
print(f"特征列表：{list(features.columns)}")

# ==============================================
# 3. 数据集划分与模型训练
# ==============================================
# 按7:3划分训练集和测试集，固定随机种子保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

print(f"\n📊 数据集划分：")
print(f"  训练集样本数：{len(X_train)}")
print(f"  测试集样本数：{len(X_test)}")

# 训练随机森林分类模型
# class_weight='balanced' 自动处理样本不平衡问题
rf_model = RandomForestClassifier(
    n_estimators=100,    # 决策树数量
    max_depth=10,        # 树最大深度，防止过拟合
    class_weight='balanced',
    random_state=42,
    n_jobs=-1            # 用全部CPU核心加速
)

print("\n⏳ 模型训练中...")
rf_model.fit(X_train, y_train)
print("✅ 模型训练完成")

# ==============================================
# 4. 模型评估
# 指标说明：
# - 准确率：整体预测正确的比例
# - 精确率：预测流失的用户里，真的流失了的比例
# - 召回率：真实流失的用户里，被我们成功找出来的比例
# - AUC：模型整体区分能力，≥0.8说明效果良好
# ==============================================
# 预测结果
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("📈 模型评估结果")
print("=" * 60)
print(f"准确率 (Accuracy)  : {accuracy:.4f} ({accuracy:.2%})")
print(f"精确率 (Precision) : {precision:.4f} ({precision:.2%})")
print(f"召回率 (Recall)    : {recall:.4f} ({recall:.2%})")
print(f"F1分数             : {f1:.4f}")
print(f"AUC值              : {auc:.4f}")
print("=" * 60)

if auc >= 0.8:
    print("✅ AUC ≥ 0.8，模型效果达标，可写入简历")
else:
    print("⚠️  AUC未达0.8，可通过调整特征或模型参数优化")

# 详细分类报告
print("\n📋 详细分类报告：")
print(classification_report(y_test, y_pred))

# ==============================================
# 5. 特征重要性排序
# 找出影响用户流失的Top5核心因素
# ==============================================
feature_importance = pd.DataFrame({
    '特征名称': features.columns,
    '重要性得分': rf_model.feature_importances_
}).sort_values('重要性得分', ascending=False)

print("\n" + "=" * 60)
print("🏆 影响用户流失的Top10特征重要性排序")
print("=" * 60)
print(feature_importance.head(10).to_string(index=False))

top5_features = feature_importance.head(5)['特征名称'].tolist()
print(f"\n📌 核心流失影响因素Top5：{', '.join(top5_features)}")

# ==============================================
# 6. 业务层面流失干预建议
# ==============================================
print("\n" + "=" * 60)
print("💡 流失干预业务建议")
print("=" * 60)
print("1. 针对活跃率低的用户：推送每日签到奖励、限时活动，提升登录频次")
print("2. 针对游戏时长短的用户：优化前期关卡体验，增加碎片化玩法，降低参与门槛")
print("3. 针对卡关用户：在关卡卡点推送攻略、道具礼包，降低挫败感")
print("4. 针对付费用户：建立专属权益体系，提升留存意愿，付费用户流失损失更大")
print("5. 针对注册7-14天的用户：重点投放七日留存礼包，渡过流失高发期")

print("\n" + "=" * 60)
print("✅ 流失预测模型构建完成！")
print("=" * 60)
