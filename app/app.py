from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and TF-IDF vectorizer
model = joblib.load('../models/spam_sms_model.pkl')
tfidf = joblib.load('../models/tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['message']
    message_transformed = tfidf.transform([message])
    message_transformed_dense = message_transformed.toarray()  # Convert to dense format
    prediction = model.predict(message_transformed_dense)
    return jsonify({'prediction': 'spam' if prediction[0] == 1 else 'ham'})

if __name__ == '__main__':
    app.run(debug=True)
