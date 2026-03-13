from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Fire_Alerts(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.String(20))
    date = db.Column(db.String(20))
    image_path = db.Column(db.String(100))

    def __repr__(self):
        return f'<FireAlert {self.id}>'