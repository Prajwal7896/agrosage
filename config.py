import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = "agro_sage_secret_key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "agro_sage.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False