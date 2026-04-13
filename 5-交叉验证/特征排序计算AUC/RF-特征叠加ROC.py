import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林模型
from sklearn.metrics import roc_auc_score

# 加载CSV文件
data = pd.read_csv('E:/RS/2分类/特征筛选/RF/RF-SHAP-特征.csv')

# 假设第一列是标签列，从第二列开始是特征列
X = data.iloc[:, 1:]  # 特征列
y = data.iloc[:, 0]  # 标签列

# 初始化结果存储
roc_results = []

# 初始化五倍交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 逐步累加特征并计算ROC值
for num_features in range(1, 24):  # 从1个特征到23个特征
    # 选择前num_features个特征
    X_selected = X.iloc[:, :num_features]

    # 存储每次交叉验证的ROC值
    fold_roc_scores = []

    # 进行五倍交叉验证
    for train_index, test_index in cv.split(X_selected, y):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 初始化随机森林模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)  # 使用100棵树

        # 训练模型
        model.fit(X_train, y_train)

        # 预测概率
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # 计算ROC值
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        fold_roc_scores.append(roc_auc)

    # 计算平均ROC值
    mean_roc_auc = np.mean(fold_roc_scores)
    roc_results.append({'Number of Features': num_features, 'ROC AUC': mean_roc_auc})

    print(f"使用前 {num_features} 个特征的平均 ROC AUC: {mean_roc_auc:.4f}")

# 将结果保存到CSV文件
roc_df = pd.DataFrame(roc_results)
roc_df.to_csv('E:/RS/2分类/特征筛选/RF/feature_roc_results_cv.csv', index=False)
print("ROC结果已保存至本地CSV文件。")