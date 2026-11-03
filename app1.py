
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="📧",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.main {
    background-color: #F7F9FC;
}
h1 {
    color: #172554;
}
h2, h3 {
    color: #1E3A8A;
}
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}
.stButton button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TRAIN MODEL ----------------

@st.cache_resource
def train_model():

    df = pd.read_csv("spam_ham_dataset.csv")

    df = df[["label", "text"]].dropna()

    df["label"] = (
        df["label"].astype(str).str.lower().str.strip()
    )

    df["text"] = df["text"].astype(str)

    df = df[df["label"].isin(["spam", "ham"])]

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english"
        )),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions,
            pos_label="spam",
            zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions,
            pos_label="spam",
            zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions,
            pos_label="spam",
            zero_division=0
        ),
        "confusion": confusion_matrix(
            y_test, predictions,
            labels=["ham", "spam"]
        )
    }

    return model, df, metrics


try:
    model, df, metrics = train_model()
except Exception as e:
    st.error(f"Model error: {e}")
    st.stop()


# ---------------- SESSION HISTORY ----------------

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------- SIDEBAR ----------------

st.sidebar.title("📧 SpamGuard AI")

st.sidebar.caption("Intelligent Email Security")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Home",
        "Email Detection",
        "Analytics",
        "Prediction History",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.success("Naive Bayes Model Active")

st.sidebar.caption(
    f"Model Accuracy: {metrics['accuracy']*100:.2f}%"
)


# ---------------- HOME DASHBOARD ----------------

if page == "Home":

    st.title("📊 SpamGuard Dashboard")

    st.write(
        "Welcome to your intelligent email classification system."
    )

    st.divider()

    total = len(df)
    spam = (df["label"] == "spam").sum()
    ham = (df["label"] == "ham").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Emails", total)
    col2.metric("Spam Emails", spam)
    col3.metric("Legitimate Emails", ham)
    col4.metric(
        "Model Accuracy",
        f"{metrics['accuracy']*100:.2f}%"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Email Distribution")

        counts = df["label"].value_counts().reindex(
            ["spam", "ham"], fill_value=0
        )

        fig, ax = plt.subplots()

        ax.pie(
            counts,
            labels=["Spam", "Legitimate"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Spam vs Legitimate Emails")

        st.pyplot(fig)

    with right:

        st.subheader("Dataset Statistics")

        chart_data = pd.DataFrame({
            "Category": ["Spam", "Legitimate"],
            "Emails": [spam, ham]
        })

        st.bar_chart(
            chart_data.set_index("Category")
        )

    st.info(
        "Dataset statistics represent the labelled dataset, "
        "not the number of emails checked by users."
    )


# ---------------- EMAIL DETECTION ----------------

elif page == "Email Detection":

    st.title("🔍 Email Spam Detection")

    st.write(
        "Enter an email message or upload a text file."
    )

    email = st.text_area(
        "Email Content",
        height=220,
        placeholder="Paste your email here..."
    )

    uploaded_file = st.file_uploader(
        "Or upload an email text file",
        type=["txt"]
    )

    if uploaded_file is not None:
        try:
            uploaded_text = uploaded_file.getvalue().decode("utf-8")
            st.caption(f"Uploaded: {uploaded_file.name}")
            email = uploaded_text
            st.text_area(
                "Uploaded email preview",
                value=email,
                height=120,
                disabled=True
            )
        except UnicodeDecodeError:
            st.error("Please upload a valid UTF-8 text file.")
            email = ""

    if st.button("Analyze Email", type="primary"):

        if not email.strip():

            st.warning("Please enter an email first.")

        else:

            prediction = model.predict([email])[0]

            probabilities = model.predict_proba([email])[0]

            classes = list(
                model.named_steps["classifier"].classes_
            )

            spam_probability = probabilities[
                classes.index("spam")
            ]

            if prediction == "spam":

                st.error("🚨 SPAM EMAIL DETECTED")

            else:

                st.success("✅ LEGITIMATE EMAIL")

            st.metric(
                "Estimated Spam Probability",
                f"{spam_probability*100:.2f}%"
            )

            st.progress(float(spam_probability))

            st.caption(
                "Probability is the model's estimate, "
                "not a guarantee of email safety."
            )

            st.session_state.history.append({
                "Email": email[:100],
                "Prediction": prediction.upper(),
                "Spam Probability": round(
                    spam_probability*100, 2
                )
            })


# ---------------- ANALYTICS ----------------

elif page == "Analytics":

    st.title("📈 Model Performance Analytics")

    st.write(
        "Evaluation results on the held-out test dataset."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{metrics['accuracy']*100:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{metrics['precision']*100:.2f}%"
    )

    c3.metric(
        "Recall",
        f"{metrics['recall']*100:.2f}%"
    )

    c4.metric(
        "F1 Score",
        f"{metrics['f1']*100:.2f}%"
    )

    st.divider()

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()

    sns.heatmap(
        metrics["confusion"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Spam"],
        yticklabels=["Legitimate", "Spam"],
        ax=ax
    )

    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Actual Category")

    st.pyplot(fig)

    st.subheader("Understanding the Evaluation")

    st.write("""
    - Accuracy: Overall correct predictions.
    - Precision: How many predicted spam emails were actually spam.
    - Recall: How many actual spam emails were detected.
    - F1 Score: Balance between precision and recall.
    """)


# ---------------- PREDICTION HISTORY ----------------

elif page == "Prediction History":

    st.title("📋 Prediction History")

    st.write(
        "View emails classified during this session."
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download History CSV",
            data=csv,
            file_name="spamguard_history.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "No email predictions yet. "
            "Go to Email Detection to get started."
        )


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.title("About SpamGuard AI")

    st.subheader("Project Objective")

    st.write("""
    SpamGuard AI is a machine learning based email
    classification system designed to identify unwanted
    spam emails and legitimate emails.
    """)

    st.subheader("Technology Stack")

    st.write("""
    1. Python
    2. Streamlit
    3. Pandas
    4. Scikit-learn
    5. TF-IDF Vectorization
    6. Multinomial Naive Bayes
    7. Matplotlib and Seaborn
    """)

    st.subheader("Project Workflow")

    st.write("""
    Dataset Collection
          ↓
    Text Preprocessing
          ↓
    TF-IDF Feature Extraction
          ↓
    Naive Bayes Training
          ↓
    Model Evaluation
          ↓
    Email Classification
    """)

    st.subheader("Machine Learning Algorithm")

    st.write("""
    Multinomial Naive Bayes is a probabilistic
    classification algorithm based on Bayes' theorem.
    It calculates the probability of an email belonging
    to a particular class using its word features.
    """)

    st.success(
        "SpamGuard AI - Intelligent Email Security System"
    )
