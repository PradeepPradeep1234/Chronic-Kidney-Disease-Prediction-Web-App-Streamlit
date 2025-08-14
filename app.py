import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.set_page_config(page_title="CKD Prediction App", layout="wide")
st.title("🩺 Chronic Kidney Disease Prediction")
st.write("Upload the **kidney_disease.csv** dataset to explore, train, and evaluate a CKD prediction model.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        
        df = pd.read_csv(uploaded_file)

        st.subheader("📌 Dataset Preview")
        st.dataframe(df.head())

        with st.expander("ℹ️ Dataset Info"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

        st.write(f"**Shape:** {df.shape}")
        st.subheader("📊 Summary Statistics")
        st.write(df.describe())

        # =================
        # Data Cleaning
        # =================
        df.columns = df.columns.str.lower()
        df = df.replace('?', np.nan)
        df = df.drop(columns=['id'], errors='ignore')

        if 'classification' not in df.columns:
            st.error("❌ 'classification' column not found in dataset.")
            st.stop()

        df['classification'] = df['classification'].astype(str).str.strip().str.lower()
        df['classification'] = df['classification'].replace({'ckd': 1, 'notckd': 0})
        df = df[df['classification'].isin([0, 1])]  # remove unknown values

        for col in df.columns:
            if col != 'classification':
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0])
                    df[col] = LabelEncoder().fit_transform(df[col])
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].mean())

        # =================
        # EDA
        # =================
        st.subheader("📊 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=True, fmt='.2f', ax=ax)
        st.pyplot(fig)

        st.subheader("📊 Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='classification', data=df, ax=ax)
        ax.set_xticklabels(['Not CKD', 'CKD'])
        st.pyplot(fig)

        # =================
        # Feature / Target Split
        # =================
        X = df.drop('classification', axis=1)
        y = df['classification']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # =================
        # Model Training
        # =================
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # =================
        # Evaluation
        # =================
        st.subheader("📈 Model Evaluation")
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        st.write(f"**Train Accuracy:** {train_acc:.2f}")
        st.write(f"**Test Accuracy:** {test_acc:.2f}")

        st.text("Classification Report:")
        st.text(classification_report(y_test, test_preds))

        st.subheader("📊 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, test_preds), annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # Overfitting / Underfitting check
        if train_acc > test_acc + 0.05:
            st.warning("⚠️ Possible Overfitting.")
        elif test_acc > train_acc:
            st.warning("⚠️ Possible Underfitting.")
        else:
            st.success("✅ Model is generalizing well.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("Please upload the dataset to proceed.")
