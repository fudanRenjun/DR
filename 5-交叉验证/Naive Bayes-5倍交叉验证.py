import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sympy.physics.control.control_plots import matplotlib

# 显示中文字体
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载CSV文件
data = pd.read_excel('E:/RS/组学/房水_处理后.xlsx')

print(data.isnull().sum())
data_filled = data.fillna(data.mean())

# 假设第一列是标签列，其他列是特征
X = data.iloc[:, 1:]  # 特征列
y = data.iloc[:, 0]  # 标签列

# 初始化模型
model = GaussianNB(var_smoothing=1e-07)

# 初始化交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 存储每次交叉验证的指标
auc_scores = []
pr_scores = []  # 新增：存储PR AUC
sensitivities = []
specificities = []
ppv_scores = []
npv_scores = []
accuracy_scores = []
f1_scores = []
fprs = []
roc_auc_scores = []
tprs = []

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # 计算各项指标
    auc_scores.append(roc_auc_score(y_test, y_pred_prob))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_scores.append(auc(recall, precision))  # 计算PR AUC

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivities.append(tp / (tp + fn))
    specificities.append(tn / (tn + fp))
    ppv_scores.append(tp / (tp + fp))
    npv_scores.append(tn / (tn + fn))
    accuracy_scores.append((tp + tn) / (tp + tn + fp + fn))
    f1_scores.append(2 * tp / (2 * tp + fp + fn))

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    # 绘制热力图
    plt.figure(figsize=(10, 5), dpi=300)
    sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 4},
                fmt='d', cmap='YlGnBu', cbar_kws={'shrink': 0.75})
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('Predicted Label', fontsize=7)
    plt.ylabel('True Label', fontsize=7)
    plt.title('Confusion Matrix Heat Map', fontsize=8)
    plt.close()

# 进行交叉验证并计算ROC曲线
for i, (train, test) in enumerate(cv.split(X, y)):
    # 训练模型
    model.fit(X.loc[train], y.loc[train])
    # 预测概率
    y_score = model.predict_proba(X.loc[test])[:, 1]
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y.loc[test], y_score)
    roc_auc = auc(fpr, tpr)
    roc_auc_scores.append(roc_auc)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, lw=1, alpha=0.5, label=f'ROC fold {i + 1} (AUC = {roc_auc:.2f})')

    # 存储TPR和FPR用于绘制平均ROC曲线
    tprs.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))
    fprs.append(np.linspace(0, 1, 100))

# 计算平均ROC曲线
mean_fpr = np.mean(fprs, axis=0)
mean_tpr = np.mean(tprs, axis=0)
mean_roc_auc = auc(mean_fpr, mean_tpr)

# 计算标准差
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

# 绘制平均ROC曲线和标准差线
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % mean_roc_auc, lw=2, alpha=.8)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

# 绘制对角线（随机猜测）
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8, label='Chance')

# 设置图例和坐标轴标签
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for 5-fold Cross-Validation')
plt.legend(loc="lower right")
plt.show()

# 输出每次交叉验证的评估指标
for i in range(5):
    print(f"第{i + 1}次交叉验证结果：")
    print(f"AUC值：{auc_scores[i]:.2f}")
    print(f"PR值：{pr_scores[i]:.2f}")  # 新增：输出PR AUC
    print(f"敏感性：{sensitivities[i]:.2f}")
    print(f"特异性：{specificities[i]:.2f}")
    print(f"阳性预测值(PPV)：{ppv_scores[i]:.2f}")
    print(f"阴性预测值(NPV)：{npv_scores[i]:.2f}")
    print(f"准确率：{accuracy_scores[i]:.2f}")
    print(f"F1得分值：{f1_scores[i]:.2f}")
    print()

# 输出最终平均结果
print("最终平均结果：")
print(f"AUC值：{np.mean(auc_scores):.2f} ± {np.std(auc_scores):.2f}")
print(f"PR值：{np.mean(pr_scores):.2f} ± {np.std(pr_scores):.2f}")  # 新增：输出平均PR AUC
print(f"敏感性：{np.mean(sensitivities):.2f} ± {np.std(sensitivities):.2f}")
print(f"特异性：{np.mean(specificities):.2f} ± {np.std(specificities):.2f}")
print(f"阳性预测值(PPV)：{np.mean(ppv_scores):.2f} ± {np.std(ppv_scores):.2f}")
print(f"阴性预测值(NPV)：{np.mean(npv_scores):.2f} ± {np.std(npv_scores):.2f}")
print(f"准确率：{np.mean(accuracy_scores):.2f} ± {np.std(accuracy_scores):.2f}")
print(f"F1得分值：{np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")

# 绘制PR曲线
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
auc_pr = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, color='blue', label='PR curve (AUC = %0.2f)' % auc_pr)
# 添加随机猜测对角线
plt.plot([0, 1], [1, 0], linestyle='--', color='red')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.show()

import shap
import pandas as pd

# 1. 初始化解释器并计算SHAP值
explainer = shap.Explainer(model.predict, X, max_evals=8000)
shap_values2 = explainer(X)

# 2. 计算平均绝对SHAP值，并按特征排序
shap_vals_array = shap_values2.values if hasattr(shap_values2, 'values') else shap_values2
mean_abs_shap = np.abs(shap_vals_array).mean(axis=0)  # 按列求平均

# 3. 构造 DataFrame
feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'MeanAbsSHAP': mean_abs_shap
})

# 4. 排序并取前100个
shap_top100 = shap_importance_df.sort_values(by='MeanAbsSHAP', ascending=False).head(100)

# 5. 保存为 Excel 文件
shap_top100.to_excel("E:/RS/组学/NB-shap_top100_features.xlsx", index=False)

# 6. 绘制前21个特征条形图
shap.summary_plot(shap_values2, X, max_display=21, plot_type="bar")
