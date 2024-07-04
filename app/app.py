from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Load the model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_sms_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')

model = joblib.load(model_path)
tfidf = joblib.load(vectorizer_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data)
        prediction = model.predict(vect)
        result = 'spam' if prediction[0] == 1 else 'ham'
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
