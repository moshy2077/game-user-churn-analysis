# 游戏用户行为分析与流失预测项目

## 项目背景
基于10万条手游用户行为数据，完成从数据清洗、业务分析到机器学习流失预测的全链路分析，挖掘用户留存与付费规律，构建流失预警模型，为游戏运营提供数据驱动的决策支持。

## 技术栈
编程语言：Python 3.12
数据处理：Pandas、NumPy
数据库：SQLite
机器学习：Scikit-learn（随机森林分类器）
可视化：Pyecharts 交互式图表
分析方法：同期群分析、转化漏斗、RFM用户分层、分类预测模型

## 项目结构
游戏用户行为分析与流失预测/
├── data/          # 原始与清洗后数据集
├── code/          # 全流程分析代码
├── output/        # 可视化图表与数据库
└── README.md      # 项目说明文档

## 运行步骤
1. 安装依赖库
```bash
pip install pandas numpy scikit-learn pyecharts openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple
2. 依次运行 `code/` 目录下的脚本，编号顺序即为执行顺序
3. 最终在 `output/` 目录查看生成的可视化看板与分析结果