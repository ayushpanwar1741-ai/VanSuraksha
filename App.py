import os

from flask import Flask
from database import db

def create_app():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    App = Flask(__name__, static_url_path='/static')
    App.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(base_dir, 'instance', 'Fire_Alerts.db').replace('\\', '/')
    db.init_app(App)
    # Import the 'view' blueprint
    from View import View
    App.register_blueprint(View, url_prefix="/")

    with App.app_context():
        db.create_all()
        print(f"Fire Alerts Database path: {App.config['SQLALCHEMY_DATABASE_URI']}")

    return App

if __name__ == '__main__':
    App = create_app()
    App.run(debug=True, port=5000, threaded=True)
