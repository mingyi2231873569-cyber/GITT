import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import copy

# ---------- 加载保存的类定义和模型 ----------
# 因为模型依赖自定义类，所以必须在此处重新定义 SuperLearnerClassifier
# （从你原来的 app.py 中复制完整定义）
try:
    from xgboost import XGBClassifier
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False

class SuperLearnerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_learners=None, meta_learner=None, cv_folds=5):
        if base_learners is None:
            self.base_learners = [
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
                ('nb', GaussianNB()),
                ('knn', KNeighborsClassifier(n_neighbors=5))
            ]
            if XGB_INSTALLED:
                try:
                    self.base_learners.append(('xgb', XGBClassifier(random_state=42)))
                except:
                    pass
        else:
            self.base_learners = base_learners

        if meta_learner is None:
            self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        else:
            self.meta_learner = meta_learner

        self.cv_folds = cv_folds
        self.is_fitted = False
        self.label_encoder = None  # 注意：预测时不需要重新编码，所以这里可以简单处理
        self.n_classes_ = None
        self.classes_ = None
        self.base_learners_final = []

    def fit(self, X, y):
        # 此方法在训练时使用，但加载模型后不会调用，因此可以留空或简单实现
        # 但为了类的完整性，这里保留一个空fit，实际训练已经在训练阶段完成
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted')
        X = check_array(X)

        # 生成基学习器的预测
        meta_features = np.zeros((X.shape[0], len(self.base_learners_final) * self.n_classes_))

        for i, (name, clf) in enumerate(self.base_learners_final):
            if hasattr(clf, 'predict_proba'):
                probas = clf.predict_proba(X)
            else:
                # 简化处理，实际应根据clf类型选择合适方法
                probas = np.ones((len(X), self.n_classes_)) / self.n_classes_
            meta_features[:, i*self.n_classes_:(i+1)*self.n_classes_] = probas

        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X):
        probas = self.predict_proba(X)
        # 注意：这里直接返回类别索引，因为加载后 label_encoder 可能未保存，所以返回整数
        return np.argmax(probas, axis=1)

# 加载模型和工具
@st.cache_resource  # 缓存模型，避免重复加载
def load_models():
    model = joblib.load('super_learner_final.joblib')
    scaler = joblib.load('scaler_final.joblib')
    feature_names = joblib.load('feature_names.joblib')
    class_names = joblib.load('class_names.joblib')
    return model, scaler, feature_names, class_names

model, scaler, feature_names, class_names = load_models()

# ---------- 页面布局 ----------
st.set_page_config(page_title="代谢物预测", layout="centered")
st.title("🧪 血浆氨基酸代谢物预测模型")
st.markdown("输入以下代谢物浓度，模型将预测属于 **健康/胃癌/结直肠癌** 的概率。")

# 创建输入表单
with st.form("input_form"):
    cols = st.columns(2)  # 分两列显示输入框
    input_values = []
    for i, feature in enumerate(feature_names):
        col = cols[i % 2]
        value = col.number_input(
            f"{feature}",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=1.0,
            format="%.2f",
            key=feature
        )
        input_values.append(value)
    
    submitted = st.form_submit_button("开始预测")

# ---------- 预测和结果显示 ----------
if submitted:
    # 将输入转换为数组并标准化
    X = np.array(input_values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # 预测
    pred_class = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]
    
    # 显示结果
    st.subheader("📊 预测结果")
    st.success(f"**预测类别：{pred_class}**")
    
    # 创建概率数据框
    prob_df = pd.DataFrame({
        '类别': class_names,
        '概率 (%)': pred_proba * 100
    })
    
    # 使用 Plotly 绘制柱状图（更美观）
    fig = go.Figure(data=[
        go.Bar(
            x=prob_df['类别'],
            y=prob_df['概率 (%)'],
            marker_color=['#2E86AB', '#A23B72', '#F18F01'],
            text=prob_df['概率 (%)'].round(1),
            textposition='outside'
        )
    ])
    fig.update_layout(
        title="各类别预测概率",
        xaxis_title="类别",
        yaxis_title="概率 (%)",
        yaxis=dict(range=[0, 100]),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 同时显示表格
    st.dataframe(prob_df, use_container_width=True)