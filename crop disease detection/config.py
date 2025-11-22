import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    # Using SQLite for simplicity. For production, use MySQL/Postgres and update SQLALCHEMY_DATABASE_URI.
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(BASE_DIR, 'plantdb.sqlite'))
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MODEL_FOLDER = os.path.join(BASE_DIR, 'model_files')
    MODEL_PATH = os.path.join(MODEL_FOLDER, 'plant_disease_model.pkl')
    LABEL_ENCODER_PATH = os.path.join(MODEL_FOLDER, 'label_encoder.pkl')