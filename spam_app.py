from IPython.display import display, Markdown
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load model and vectorizer
with open("dt_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Define stopwords
stop_words = set(stopwords.words('english'))

# Styled input prompt
display(Markdown("## 💬 Spam Message Checker"))
display(Markdown("Please type your message below to classify it as **SPAM** or **VALID**."))

# Input message
user_message = input("👉 Enter your message here: ")

# Clean the message
def clean_input_message(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Process and predict
cleaned = clean_input_message(user_message)
transformed = vectorizer.transform([cleaned])
prediction = model.predict(transformed)[0]

# Show result
if prediction.lower() == 'spam':
    display(Markdown("### 🔴 **This message is classified as: SPAM 🚫**"))
else:
    display(Markdown("### 🟢 **This message is classified as: VALID ✅**"))

# Explanation
display(Markdown("""
---
### 🤖 How It Works
This message was classified using a **Decision Tree** model trained on labeled email data.

- Text is cleaned, tokenized, and vectorized using **TF-IDF**.
- The Decision Tree applies learned rules to decide whether the message is spam or valid.
- It offers clear decision paths and strong performance in detecting spam.
"""))


        - Preprocessing includes text cleaning, tokenization, stop word removal, and TF-IDF vectorization.
        - Naive Bayes then uses learned word patterns to predict the category.
        - It’s a fast, lightweight approach commonly used in email spam filters.
        """)

        st.markdown("---")
        st.caption("Built for MIS 542 — Spam Classifier Project by Team 2")
