from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

app = Flask(__name__)
app.secret_key = "secret123"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///plant_app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ------------------ MODELS ------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(200))

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    predicted_disease = db.Column(db.String(150))
    image_path = db.Column(db.String(300))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))

# ------------------ DB INIT ------------------
with app.app_context():
    db.create_all()

# ------------------ LOGIN ------------------
@app.route("/", methods=["GET"])
@app.route("/login", methods=["GET"])
def login():
    email = request.args.get("email")
    password = request.args.get("password")

    if email and password:
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            return redirect("/history")
        else:
            return "Invalid credentials"

    return render_template("login.html")

# ------------------ SIGNUP ------------------
@app.route("/signup", methods=["GET"])
def signup():
    full_name = request.args.get("full_name")
    email = request.args.get("email")
    password = request.args.get("password")

    if full_name and email and password:
        if User.query.filter_by(email=email).first():
            return "User already exists"

        user = User(
            full_name=full_name,
            email=email,
            password=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        return redirect("/login")

    return render_template("signup.html")

# ------------------ HISTORY ------------------
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect("/login")

    records = PredictionHistory.query.filter_by(
        user_id=session["user_id"]
    ).all()

    return render_template("history.html", records=records)

# ------------------ LOGOUT ------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

if __name__ == "__main__":
    app.run(debug=True)
