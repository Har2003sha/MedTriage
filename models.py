from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="patient")  # patient | staff | admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    triage_records = db.relationship(
        "TriageRecord", backref="patient", lazy=True, cascade="all, delete-orphan"
    )


class TriageRecord(db.Model):
    __tablename__ = "triage_records"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    language = db.Column(db.String(10), nullable=False)   # en, hi, mr, ta, te
    raw_symptom_text = db.Column(db.Text, nullable=False)

    urgency_level = db.Column(db.String(20), nullable=False)     # Emergency/Urgent/Non-Urgent
    urgency_score = db.Column(db.Integer, nullable=False)        # numeric 1-10
    department = db.Column(db.String(80), nullable=False)
    matched_symptoms = db.Column(db.Text)                        # comma separated, in English
    ai_summary = db.Column(db.Text)
    summary_source = db.Column(db.String(20), default="template")  # "llm" or "template"

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "language": self.language,
            "raw_symptom_text": self.raw_symptom_text,
            "urgency_level": self.urgency_level,
            "urgency_score": self.urgency_score,
            "department": self.department,
            "matched_symptoms": self.matched_symptoms,
            "ai_summary": self.ai_summary,
            "summary_source": self.summary_source,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }
