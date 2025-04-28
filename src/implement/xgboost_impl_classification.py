import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ..utils.get_dummies import get_dummies,get_bool
from ..utils.drop_year_data import clear_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from collections import Counter


def solve_data(df:pd.DataFrame):
    true_false_bool = {"是":1, "否":0}
    gender_bool = {"男":1,"女":0}
    df["鉴定结果大类"].fillna("/", inplace=True)
    df["确诊食源性疾病"] = df["鉴定结果大类"].apply(lambda x: 0 if x == "/" or x == None else 1)
    df.drop(columns=["鉴定结果大类"], inplace=True)
    df.drop(columns=['鉴定结论'], inplace=True)
    df.drop(columns=["是否复诊"], inplace=True)
    df.drop(columns=["是否住院"], inplace=True)
    df = get_bool(df,"其他人是否发病",true_false_bool)
    df = get_bool(df,"患者性别",gender_bool)
    df.drop(columns=["现在住址地市"], inplace=True)
    df = get_dummies(["食品分类", "加工及包装方式", "进食场所类型"],df)
    df = clear_data(df,"发病日期", 2023)
    df["发病月份"] = pd.to_datetime(df["发病日期"]).dt.month
    df.drop(columns=["发病日期"], inplace=True)
    return df


def to_dataset(df:pd.DataFrame) -> Tuple:
    X = df.drop(columns=['确诊食源性疾病'])
    y = df['确诊食源性疾病']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray):
    scale_pos_weight = len(y_train) / (2 * Counter(y_train)[1])
    model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False,
    )

    # 5. Predict and evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return model


def get_importance(model):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题
    importance = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values(by='Importance', ascending=False).head(15)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], color='skyblue')
    plt.xlabel('Importance (Gain)')
    plt.title('前15最重要的特征')
    plt.tight_layout()
    plt.show()

def plot_heap(X_test, y_test, model):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc_auc(X_test, y_test, model):
    y_probs = model.predict_proba(X_test)[:, 1]  # use model.predict(dtest) if using xgb.train

    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
