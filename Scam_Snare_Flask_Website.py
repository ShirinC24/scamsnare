from flask import Flask, render_template, request, session
import pickle
import nltk
from nltk.corpus import stopwords
import string
import json

app = Flask(__name__)
app.secret_key = "scamsnare_secret_2024"

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Common scam trigger words to highlight
SCAM_WORDS = [
    "free", "winner", "won", "prize", "claim", "urgent", "act now", "limited time",
    "click here", "verify", "account", "suspended", "password", "bank", "credit",
    "offer", "congratulations", "selected", "reward", "cash", "lottery", "gift",
    "guaranteed", "risk-free", "million", "billion", "inheritance", "transfer",
    "otp", "pin", "cvv", "ssn", "social security", "tax", "refund", "irs",
    "amazon", "paypal", "bitcoin", "crypto", "investment", "double", "profit",
    "call now", "reply", "confirm", "update", "expire", "immediately", "alert"
]

def preprocess(text):
    nltk.download('stopwords', quiet=True)
    stop_words = stopwords.words('english')
    text_clean = text.lower()
    text_clean = "".join([c for c in text_clean if c not in string.punctuation])
    text_clean = " ".join([word for word in text_clean.split() if word not in stop_words])
    return text_clean

def highlight_scam_words(message):
    """Wrap scam trigger words in <mark> tags for highlighting."""
    highlighted = message
    for word in sorted(SCAM_WORDS, key=len, reverse=True):
        import re
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(f'<mark class="scam-word">{word}</mark>', highlighted)
    return highlighted

@app.route("/")
def home():
    history = session.get("history", [])
    return render_template("index.html", history=history)

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    cleaned = preprocess(message)
    transformed = vectorizer.transform([cleaned])

    result = model.predict(transformed)[0]
    proba = model.predict_proba(transformed)[0]

    if result == 1:
        confidence = round(proba[1] * 100, 1)
        prediction = "🚨 Scam Message Detected!"
        is_scam = True
        risk_level = "HIGH" if confidence > 80 else "MEDIUM"
    else:
        confidence = round(proba[0] * 100, 1)
        prediction = "✅ Safe Message"
        is_scam = False
        risk_level = "LOW"

    highlighted_message = highlight_scam_words(message) if is_scam else message

    # Save to session history (keep last 10)
    history = session.get("history", [])
    history.insert(0, {
        "message": message[:80] + ("..." if len(message) > 80 else ""),
        "is_scam": is_scam,
        "confidence": confidence,
        "risk_level": risk_level
    })
    session["history"] = history[:10]

    return render_template(
        "index.html",
        prediction=prediction,
        user_message=message,
        highlighted_message=highlighted_message,
        is_scam=is_scam,
        confidence=confidence,
        risk_level=risk_level,
        history=session.get("history", [])
    )

@app.route("/clear-history", methods=["POST"])
def clear_history():
    session.pop("history", None)
    return ("", 204)

if __name__ == "__main__":
    nltk.download('stopwords', quiet=True)
    app.run(host='127.0.0.1', port=5000, debug=True)
