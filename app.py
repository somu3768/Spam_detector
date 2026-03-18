from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        email = request.form.get("email")

        data = vectorizer.transform([email])
        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "Spam ❌"
        else:
            result = "Not Spam ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        print(e)
        return render_template("index.html", prediction_text="Error occurred")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)