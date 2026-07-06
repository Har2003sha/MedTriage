from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length

from models import db, TriageRecord
from models_ml.classifier import score_symptoms, SUPPORTED_LANGUAGES
from llm_service import generate_summary

triage_bp = Blueprint("triage", __name__)


class TriageForm(FlaskForm):
    language = SelectField(
        "Language",
        choices=[(code, name) for code, name in SUPPORTED_LANGUAGES.items()],
        validators=[DataRequired()],
    )
    symptom_text = TextAreaField(
        "Describe your symptoms",
        validators=[DataRequired(), Length(min=3, max=2000)],
    )
    submit = SubmitField("Get Triage Assessment")


@triage_bp.route("/")
@login_required
def dashboard():
    records = (
        TriageRecord.query.filter_by(user_id=current_user.id)
        .order_by(TriageRecord.created_at.desc())
        .limit(10)
        .all()
    )
    return render_template("dashboard.html", records=records)


@triage_bp.route("/triage/new", methods=["GET", "POST"])
@login_required
def new_triage():
    form = TriageForm()
    if form.validate_on_submit():
        language = form.language.data
        text = form.symptom_text.data.strip()

        result = score_symptoms(text, language)

        summary, source = generate_summary(
            raw_text=text,
            language=language,
            urgency=result["urgency"],
            score=result["score"],
            department=result["department"],
            matched_terms=result["matched_terms"],
        )

        record = TriageRecord(
            user_id=current_user.id,
            language=language,
            raw_symptom_text=text,
            urgency_level=result["urgency"],
            urgency_score=result["score"],
            department=result["department"],
            matched_symptoms=", ".join(result["matched_terms"]),
            ai_summary=summary,
            summary_source=source,
        )
        db.session.add(record)
        db.session.commit()

        return redirect(url_for("triage.result", record_id=record.id))

    return render_template("triage_form.html", form=form)


@triage_bp.route("/triage/<int:record_id>")
@login_required
def result(record_id):
    record = TriageRecord.query.filter_by(id=record_id, user_id=current_user.id).first_or_404()
    return render_template("triage_result.html", record=record)
