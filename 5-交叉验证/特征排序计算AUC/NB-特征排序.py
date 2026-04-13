import numpy as np
import pandas as pd
import shap
from sklearn.naive_bayes import GaussianNB

# 加载CSV文件
data = pd.read_excel('E:/RS/2分类/训练集.xlsx')

# 填充缺失值
data_filled = data.fillna(data.mean())

# 假设第一列是标签列，其他列是特征
X = data.iloc[:, 5:]  # 特征列
y = data.iloc[:, 0]   # 标签列

# 初始化朴素贝叶斯模型
model = GaussianNB()

# 训练模型（使用全部数据）
model.fit(X, y)

# 使用KernelExplainer计算SHAP值
explainer = shap.KernelExplainer(model.predict_proba, X)
shap_values = explainer.shap_values(X)

# 将SHAP值转换为Explanation对象
shap_values2 = shap.Explanation(
    values=shap_values[1],  # 取正类（类别1）的SHAP值
    base_values=explainer.expected_value[1],  # 取正类的基准值
    data=X.values,
    feature_names=X.columns
)

# 绘制所有特征的条形图（beeswarm图）
shap.plots.beeswarm(shap_values2, order=shap_values2.abs.max(0), max_display=23)

# 获取特征重要性排序
feature_importance = np.abs(shap_values2.values).mean(axis=0)
sorted_idx = np.argsort(feature_importance)[::-1]  # 从高到低排序
sorted_features = X.columns[sorted_idx]  # 按重要性排序的特征名

# 按特征重要性从高到低对原始数据进行排序
sorted_data = data.iloc[:, np.concatenate([[0], sorted_idx + 5])]  # 保留标签列和排序后的特征列

# 保存排序后的数据到本地
sorted_data.to_csv('E:/RS/2分类/特征筛选/NB/sorted_data_by_shap.csv', index=False)
print("按特征重要性排序后的数据已保存至本地CSV文件。")