import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# 1. 数据加载与清洗
data = pd.read_excel('E:/RS/LZZ/LZZ-xun.xlsx')
label_col = data.columns[0]
data = data.dropna(subset=[label_col])
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# 2. 定义 5 折 CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# 3. 定义 Optuna 目标函数：返回在 5 折 CV 上的平均 AUC
def objective(trial):
    # 超参数搜索空间
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }

    aucs = []
    for train_idx, valid_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, prob))

    # 返回平均 AUC，Optuna 会自动最大化它
    return np.mean(aucs)


# 4. 创建并运行 Study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

# 5. 输出最佳参数和对应的 CV 平均 AUC
print("Best parameters:")
for key, val in study.best_trial.params.items():
    print(f"  {key}: {val}")
print(f"\nBest 5-fold CV AUC: {study.best_value:.4f}")

# （可选）使用最佳参数训练全数据集，并保存模型
best_params = study.best_trial.params
final_model = XGBClassifier(**best_params,
                            use_label_encoder=False,
                            eval_metric='logloss',
                            random_state=42)
final_model.fit(X, y)
# e.g., pickle.dump(final_model, open('xgb_final.pkl', 'wb'))

