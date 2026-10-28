
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Website settings
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Website design
st.markdown("""
<style>
.stApp {
    background-color: #f2f6fc;
}
.title {
    text-align: center;
    color: #173b68;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #536579;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_resource
def train_model():
    data = pd.read_csv(
        "data/SMSSpamCollection",
        sep="\t",
        names=["label", "message"],
        encoding="utf-8"
    )

    # Convert spam and ham into numbers
    data["label"] = data["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Separate messages and labels
    X = data["message"]
    y = data["label"]

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Convert text into numerical features
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Train SVM model
    model = LinearSVC()
    model.fit(X_train, y_train)

    # Calculate test accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return vectorizer, model, accuracy


# Train the model
try:
    vectorizer, model, accuracy = train_model()
except Exception as e:
    st.error("Could not load the dataset. Check the data folder.")
    st.stop()

# Frontend design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef4ff, #f8fbff);
    color: #17345c;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(120deg, #102447, #244b85);
    padding: 35px 25px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(20, 45, 85, 0.15);
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    color: #d5e5ff;
    font-size: 16px;
    margin-bottom: 0;
}

.stTextArea textarea {
    background-color: white;
    color: #17345c;
    border: 1px solid #cbdaf0;
    border-radius: 12px;
    font-size: 16px;
}

.stTextArea textarea:focus {
    border-color: #3578e5;
    box-shadow: 0 0 0 2px #d7e6ff;
}

.stFormSubmitButton button {
    background: linear-gradient(90deg, #1677e8, #3155c8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: bold;
    min-height: 48px;
    transition: 0.2s;
}

.stFormSubmitButton button:hover {
    background: #103f9e;
    color: white;
    border: none;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stExpander"] {
    background-color: white;
    border: 1px solid #d9e5f7;
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #7185a5;
    font-size: 13px;
    padding-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hero">
    <h1>SMS Spam Detector</h1>
    <p>Check whether a message is Spam or Not Spam</p>
</div>
""", unsafe_allow_html=True)

# Main message card
st.info(
    "Enter an SMS message below. "
    "Our machine learning model will analyze its text."
)

with st.form("message_form"):
    message = st.text_area(
        "Enter your SMS message",
        placeholder=(
            "Example: Congratulations! "
            "You won a free prize."
        ),
        height=150
    )

    submitted = st.form_submit_button(
        "Check Message",
        use_container_width=True
    )

# Prediction result
if submitted:
    if not message.strip():
        st.warning("Please enter a message before checking.")
    else:
        message_data = vectorizer.transform([message])
        result = model.predict(message_data)[0]

        if result == 1:
            st.error("🚨 SPAM MESSAGE DETECTED")
            st.write(
                "This message resembles spam messages "
            )
        else:
            st.success("✅ THIS MESSAGE IS NOT SPAM")
            st.write(
                "This message resembles legitimate messages "
            )


# Footer
st.markdown("""
<div class="footer">
    SMS Spam Detection
</div>
""", unsafe_allow_html=True)
