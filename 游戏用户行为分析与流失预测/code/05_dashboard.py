# ==============================================
# 阶段6：交互式数据看板
# 整合4大核心分析板块，输出dashboard.html
# ==============================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pyecharts.charts import Line, Pie, Bar, Page
from pyecharts import options as opts
from pyecharts.globals import ThemeType

# --------------------------
# 0. 数据读取与基础预处理
# --------------------------
df = pd.read_csv("cleaned_game_data.csv", encoding="utf-8-sig")
df['register_date'] = pd.to_datetime(df['register_date'])
df['last_login_date'] = pd.to_datetime(df['last_login_date'])
analysis_date = pd.to_datetime('2026-11-20')

print("=" * 60)
print("📊 正在生成交互式数据看板...")
print("=" * 60)

# ==============================================
# 图表1：用户新增活跃趋势图
# 展示按周统计的新用户注册趋势，反映产品拉新效果
# ==============================================
# 按日期统计每日新增用户，按周聚合让图表更清晰
daily_register = df.groupby(df['register_date'].dt.date).size().reset_index(name='新增用户数')
daily_register['register_date'] = pd.to_datetime(daily_register['register_date'])
weekly_register = daily_register.resample('W', on='register_date').sum().reset_index()

chart_trend = (
    Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="400px"))
    .add_xaxis(weekly_register['register_date'].dt.strftime('%Y-%m-%d').tolist())
    .add_yaxis(
        "周新增用户数",
        weekly_register['新增用户数'].tolist(),
        is_smooth=True,
        linestyle_opts=opts.LineStyleOpts(width=3),
        label_opts=opts.LabelOpts(is_show=False)
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="用户新增活跃趋势", subtitle="按周统计注册用户规模"),
        xaxis_opts=opts.AxisOpts(name="日期", axislabel_opts=opts.LabelOpts(rotate=30)),
        yaxis_opts=opts.AxisOpts(name="用户数量"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(is_show=False)
    )
)

# ==============================================
# 图表2：同期群留存曲线图
# 展示用户注册后1-7日的平均留存衰减曲线
# ==============================================
df['life_days'] = (df['last_login_date'] - df['register_date']).dt.days
# 只统计注册满7天的用户，保证计算公平
cohort_valid = df[df['register_date'] <= analysis_date - pd.Timedelta(days=7)]

retention_rates = []
day_labels = []
for day in range(1, 8):
    rate = (cohort_valid['life_days'] >= day).mean() * 100
    retention_rates.append(round(rate, 2))
    day_labels.append(f"第{day}日")

chart_retention = (
    Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="400px"))
    .add_xaxis(day_labels)
    .add_yaxis(
        "平均留存率",
        retention_rates,
        is_smooth=True,
        linestyle_opts=opts.LineStyleOpts(width=3, color="#61a0a8"),
        itemstyle_opts=opts.ItemStyleOpts(color="#61a0a8"),
        label_opts=opts.LabelOpts(is_show=True, formatter="{c}%")
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="用户同期群留存曲线", subtitle="注册用户1-7日留存表现"),
        xaxis_opts=opts.AxisOpts(name="注册后天数"),
        yaxis_opts=opts.AxisOpts(name="留存率(%)"),
        tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}%"),
        legend_opts=opts.LegendOpts(is_show=False)
    )
)

# ==============================================
# 图表3：用户付费分布饼图
# 展示不同付费能力的用户占比，反映收入结构
# ==============================================
# 划分付费层级
def get_pay_level(amount):
    if amount <= 0:
        return "未付费用户"
    elif amount <= 50:
        return "小额付费(0-50元)"
    elif amount <= 200:
        return "中额付费(50-200元)"
    else:
        return "大额付费(200元以上)"

df['pay_level'] = df['total_pay_amount'].apply(get_pay_level)
pay_stats = df['pay_level'].value_counts().reset_index()
pay_stats.columns = ['付费层级', '用户数']

chart_pay = (
    Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="400px"))
    .add(
        "",
        list(zip(pay_stats['付费层级'], pay_stats['用户数'])),
        radius=["40%", "70%"],
        label_opts=opts.LabelOpts(formatter="{b}: {d}%")
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="用户付费分层分布"),
        legend_opts=opts.LegendOpts(pos_left="left", orient="vertical")
    )
    .set_series_opts(
        tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}人 ({d}%)")
    )
)

# ==============================================
# 图表4：用户流失风险等级分布图
# 基于流失预测模型，将用户分为低/中/高三个风险等级
# ==============================================
# 复用阶段5的特征工程与模型逻辑
df['register_days'] = (analysis_date - df['register_date']).dt.days

# 构建特征
features = pd.DataFrame({
    'total_login_days': df['total_login_days'],
    'avg_daily_minutes': df['avg_daily_minutes'],
    'level_passed': df['level_passed'],
    'user_level': df['user_level'],
    'tutorial_finished': df['tutorial_finished'],
    'is_paid': df['is_paid'],
    'total_pay_amount': df['total_pay_amount'],
    'register_days': df['register_days'],
    'active_rate': df['total_login_days'] / df['register_days'],
    'avg_level_per_day': df['level_passed'] / df['total_login_days'],
    'pay_intensity': df['total_pay_amount'] / (df['total_login_days'] + 1)
})
device_dummies = pd.get_dummies(df['device_type'], prefix='device', drop_first=True)
features = pd.concat([features, device_dummies], axis=1)
features = features.replace([np.inf, -np.inf], 0).fillna(0)

# 训练模型并预测全量用户流失概率
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
df['is_churn'] = ((analysis_date - df['last_login_date']).dt.days >= 7).astype(int)
rf_model.fit(features, df['is_churn'])
df['churn_prob'] = rf_model.predict_proba(features)[:, 1]

# 划分风险等级
def get_risk_level(prob):
    if prob < 0.3:
        return "低风险"
    elif prob < 0.7:
        return "中风险"
    else:
        return "高风险"

df['risk_level'] = df['churn_prob'].apply(get_risk_level)
risk_order = ["低风险", "中风险", "高风险"]
risk_stats = df['risk_level'].value_counts().reindex(risk_order).reset_index()
risk_stats.columns = ['风险等级', '用户数']

chart_risk = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="400px"))
    .add_xaxis(risk_stats['风险等级'].tolist())
    .add_yaxis(
        "用户数",
        risk_stats['用户数'].tolist(),
        itemstyle_opts=opts.ItemStyleOpts(color="#c23531"),
        label_opts=opts.LabelOpts(is_show=True, position="top")
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="用户流失风险等级分布"),
        xaxis_opts=opts.AxisOpts(name="风险等级"),
        yaxis_opts=opts.AxisOpts(name="用户数量"),
        tooltip_opts=opts.TooltipOpts(formatter="{b}: {c}人"),
        legend_opts=opts.LegendOpts(is_show=False)
    )
)

# ==============================================
# 组合所有图表，生成完整看板
# ==============================================
# SimplePageLayout 采用上下流式布局，自动适配宽度
page = Page(layout=Page.SimplePageLayout, page_title="游戏用户行为分析看板")
page.add(chart_trend, chart_retention, chart_pay, chart_risk)
page.render("dashboard.html")

print("\n" + "=" * 60)
print("✅ 交互式数据看板生成完成！")
print("=" * 60)
print("📄 输出文件：dashboard.html")
print("🖱️  查看方式：双击文件，用Chrome/Edge浏览器打开")
print("📊 包含板块：用户新增趋势、留存曲线、付费分布、流失风险分布")
print("✨ 交互功能：鼠标悬停查看详情、图例筛选、自适应缩放")
print("=" * 60)
