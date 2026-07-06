from flask import Flask
from flask_migrate import Migrate

from config import Config
from models import db, User
from extensions import bcrypt, login_manager
from auth import auth_bp
from triage import triage_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    Migrate(app, db)

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(triage_bp, url_prefix="")

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    with app.app_context():
        db.create_all()

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
