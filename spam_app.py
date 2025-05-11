import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder

def clean_message(text):
    """
    Clean and preprocess the message text.
    """
    text = text.lower()  # Lowercase the text
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove hyperlinks
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespaces
    tokens = text.split()  # Split into tokens
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]  # Remove stopwords
    return ' '.join(tokens)

# Load model and vectorizer
with open("best_model.pkl", "rb") as model_file:  # Assuming you saved the best tuned model
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# --------- Streamlit App UI ---------
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

st.title("📧 Spam Message Classifier")
st.write("Welcome! This app detects whether a message is SPAM or VALID using a machine learning model.")

st.markdown("---")

# Input text box
user_input = st.text_area("✍️ Write your message below:", height=150, placeholder="e.g. Congratulations! You’ve won a $1,000 gift card...")

if st.button("🔍 Check Message"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Clean the input message
        cleaned_text = clean_message(user_input)
        
        # Transform the cleaned message using the vectorizer
        transformed_text = vectorizer.transform([cleaned_text])
        
        # Make the prediction
        prediction = model.predict(transformed_text)[0]

        # Display the result
        st.markdown("### 📊 Result:")
        if prediction.lower() == "spam":
            st.error("🔴 This message is SPAM! 🚫")
        else:
            st.success("🟢 This message is VALID! ✅")

        # Display the explanation of the model
        st.markdown("---")
        st.markdown("### 🤖 Model Explanation")
        st.info("""
        This model is based on the best-performing classifier (e.g., Decision Tree, Naive Bayes, or KNN) tuned for spam detection.

        - Preprocessing includes text cleaning, tokenization, stop word removal, and TF-IDF vectorization.
        - The model uses learned patterns from training data to predict whether a message is SPAM or VALID.
        - This approach is optimized to minimize false positives (valid messages marked as spam) and false negatives (spam messages not detected).
        """)

        st.markdown("---")
        st.caption("Built for MIS 542 — Spam Classifier Project by Team 2")
