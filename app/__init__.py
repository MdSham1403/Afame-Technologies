from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import routes here to avoid circular imports
    from .app import home, predict
    app.add_url_rule('/', view_func=home)
    app.add_url_rule('/predict', view_func=predict, methods=['POST'])

    return app
