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

# ---------- 导入可能的 XGBoost ----------
try:
    from xgboost import XGBClassifier
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False

# ---------- SuperLearnerClassifier 类定义（请从你原来的 app.py 完整复制）----------
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

# ---------- 加载模型和工具 ----------
@st.cache_resource
def load_models():
    model = joblib.load('super_learner_final.joblib')
    scaler = joblib.load('scaler_final.joblib')
    feature_names = joblib.load('feature_names.joblib')
    class_names = joblib.load('class_names.joblib')
    return model, scaler, feature_names, class_names

model, scaler, feature_names, class_names = load_models()

# ---------- 特征显示名称和单位映射 ----------
# 注意：feature_names 是从文件加载的原始名称（如 'Goose_deoxycholic_acid'）
# 我们需要将其映射为显示名称，并加上单位
display_names = {
    'phenylalanine': 'Phenylalanine',
    'Goose_deoxycholic_acid': 'Chenodeoxycholic acid',   # 按用户要求修改
    'Glycine': 'Glycine',
    'Glutamine': 'Glutamine',
    'Citrulline': 'Citrulline',
    'Arginine': 'Arginine',
    'Tyrosine': 'Tyrosine',
    'Leucine': 'Leucine',
    'Proline': 'Proline',
    'Serine': 'Serine',
    'Threonine': 'Threonine',
    'Asparagine': 'Asparagine',
    'Valine': 'Valine',
    'Isoleucine': 'Isoleucine',
    'BCAA_AAA': 'BCAA/AAA',   # 按用户要求修改
}

# 单位映射
units = {
    'phenylalanine': 'μmol/L',
    'Goose_deoxycholic_acid': 'nmol/ml',
    'Glycine': 'μmol/L',
    'Glutamine': 'μmol/L',
    'Citrulline': 'μmol/L',
    'Arginine': 'μmol/L',
    'Tyrosine': 'μmol/L',
    'Leucine': 'μmol/L',
    'Proline': 'μmol/L',
    'Serine': 'μmol/L',
    'Threonine': 'μmol/L',
    'Asparagine': 'μmol/L',
    'Valine': 'μmol/L',
    'Isoleucine': 'μmol/L',
    'BCAA_AAA': '',  # 无单位
}

# 生成用于显示的标签列表（保持与 feature_names 顺序一致）
labels = []
for fname in feature_names:
    base = display_names.get(fname, fname)
    unit = units.get(fname, '')
    if unit:
        label = f"{base} ({unit})"
    else:
        label = base
    labels.append(label)

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="Metabolite Prediction Model",
    page_icon="🧪",
    layout="centered"
)

st.title("🧪 Plasma Amino Acid Metabolite Prediction Model")
st.markdown("Enter the concentrations of the following metabolites to predict the probability of **Healthy / Gastric Cancer / Colorectal Cancer**.")

# ---------- 输入表单 ----------
with st.form("input_form"):
    cols = st.columns(2)
    input_values = []
    for i, label in enumerate(labels):
        col = cols[i % 2]
        # 设置默认值，可根据实际情况调整
        val = col.number_input(
            label,
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=1.0,
            format="%.2f",
            key=f"feat_{i}"
        )
        input_values.append(val)
    
    submitted = st.form_submit_button("Predict")

# ---------- 预测和结果显示 ----------
if submitted:
    X = np.array(input_values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # 模型预测（可能返回索引，也可能返回名称，这里统一处理）
    pred_result = model.predict(X_scaled)[0]
    # 如果结果是数字索引，则转换为名称
    if isinstance(pred_result, (int, np.integer)):
        pred_class = class_names[pred_result]
    else:
        pred_class = pred_result
    
    pred_proba = model.predict_proba(X_scaled)[0]
    
    # 显示结果
    st.subheader("📊 Prediction Result")
    st.success(f"**Diagnosis: {pred_class}**")
    
    # 概率柱状图
    prob_df = pd.DataFrame({
        'Class': class_names,
        'Probability (%)': pred_proba * 100
    })
    
    fig = go.Figure(data=[
        go.Bar(
            x=prob_df['Class'],
            y=prob_df['Probability (%)'],
            marker_color=['#2E86AB', '#A23B72', '#F18F01'],
            text=prob_df['Probability (%)'].round(1),
            textposition='outside'
        )
    ])
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 概率表格
    st.dataframe(prob_df, use_container_width=True)

# ---------- 底部免责声明和作者信息 ----------
st.markdown("---")
st.markdown(
    """
    **Disclaimer**: This tool is for research purposes only. It is based on a retrospective study and has not been validated for clinical use. Results should not be used as the sole basis for diagnosis or treatment decisions.

    **Author Information**: Xiao-hua Jiang, Shun Zhang, Ming-yi Yuan. Department of Gastrointestinal Surgery, Shanghai East Hospital, School of Medicine, Tongji University.
    """
)