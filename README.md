# Afame Technologies - Spam SMS Detection

This repository contains the code and resources for the Spam SMS Detection project. The project aims to build a machine learning model that can classify SMS messages as spam or legitimate.

## Project Structure

- `data/`: Contains the dataset.
- `notebooks/`: Jupyter notebooks for different stages of the project.
- `models/`: Saved models and vectorizers.
- `app/`: Flask application for deploying the model.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.
- `.gitignore`: Git ignore file.

## Setup Instructions
1. Clone the repository:
   ```bash

   git clone https://github.com/your-username/spam-sms-detection.git
   cd spam-sms-detection

2. Create and activate a Conda environment:
   ```bash
   conda create --name spam_sms_detection python=3.9
   conda activate spam_sms_detection

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Download NLTK resources (optional): If you're using NLTK for text preprocessing (e.g., stopword removal), run:
   ```python
   import nltk
   nltk.download('stopwords')

5. Run Jupyter notebooks: Navigate to the notebooks/ directory and open the notebooks in Jupyter:
   ```bash
   jupyter notebook

6. Run the Flask app: Navigate to the app/ directory and start the Flask application:
   ```bash
   cd app
   flask run

7. Access the web interface: Open your browser and go to http://127.0.0.1:5000/ to use the Spam SMS Detection app.
   
