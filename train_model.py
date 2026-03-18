import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

data = {
    "message":[
        "Win money now",
        "Free lottery ticket",
        "Hello how are you",
        "Let's meet tomorrow",
        "Congratulations you won prize",
        "Call me later"
    ],
    "label":[1,1,0,0,1,0]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])

model = MultinomialNB()
model.fit(X,df["label"])

pickle.dump(model,open("model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

print("Model Trained Successfully")