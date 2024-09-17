from flask import render_template, request, jsonify
import joblib
import os

# Routes
def home():
    return render_template('index.html')

def predict():
    try:
        message = request.form['message']
        data = [message]

        # Load vectorizer and model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_sms_model.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')
        
        model = joblib.load(model_path)
        tfidf = joblib.load(vectorizer_path)

        # Process message
        vect = tfidf.transform(data)
        vect_dense = vect.toarray()

        # Predict
        prediction = model.predict(vect_dense)
        result = 'spam' if prediction[0] == 1 else 'ham'

        return render_template('index.html', prediction=result)
    except Exception as e:
        return jsonify(error=str(e)), 500
