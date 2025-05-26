import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Page Config
st.set_page_config(page_title="Gut Microbiome Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Static Data Visualization", "Model Evaluation"])

# Static CSV Path
STATIC_CSV_PATH = "gut_microbiome_cancer_dataset.csv"

st.title("Gut Microbiome Signatures in Cancer: A Machine Learning Approach")

if page == "Static Data Visualization":
    st.header("Static Data Visualization")
    df = pd.read_csv(STATIC_CSV_PATH)
    st.dataframe(df.head())

    st.subheader("Bar Chart Visualization")
    x_col = st.selectbox("Select X-axis column", df.columns)
    y_col = st.selectbox("Select Y-axis column (Numeric)", df.select_dtypes(include='number').columns)

    if st.button("Generate Bar Chart"):
        fig, ax = plt.subplots()
        sns.barplot(x=x_col, y=y_col, data=df, ax=ax)
        st.pyplot(fig)

elif page == "Model Evaluation":
    st.header("Model Evaluation")

    uploaded_train = st.file_uploader("Upload Training CSV", type=["csv"], key="train")
    uploaded_test = st.file_uploader("Upload Testing CSV", type=["csv"], key="test")

    if uploaded_train is not None and uploaded_test is not None:
        train_df = pd.read_csv(uploaded_train)
        test_df = pd.read_csv(uploaded_test)

        # Encode target labels
        label_encoder = LabelEncoder()
        train_df["Treatment Response"] = label_encoder.fit_transform(train_df["Treatment Response"])
        test_df["Treatment Response"] = label_encoder.transform(test_df["Treatment Response"])

        label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
        st.write("Label Mapping:", label_mapping)

        features = ["Bacteroides", "Fusobacteria", "Proteobacteria", "Alpha Diversity", "Age"]
        target = "Treatment Response"

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc

        st.subheader("Model Accuracy Comparison")
        st.bar_chart(pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']))

        selected_model = st.selectbox("Select Model to View Report", list(models.keys()))
        if selected_model:
            model = models[selected_model]
            y_pred = model.predict(X_test)
            st.subheader(f"Classification Report: {selected_model}")
            st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

st.sidebar.markdown("---")
st.sidebar.caption("Created by Saloni Sharma")
