"""
Generates a clinical-style triage summary for hospital staff.

No external API / API key required — this is a fully offline, rule-based
natural-language generator built from the classifier's output
(urgency, department, matched symptom categories). It always works,
with zero setup and zero external calls.

(If you later want richer AI-written summaries, you can plug in any LLM
here — this function's signature/return shape is designed to make that a
drop-in change. But it is NOT required for the app to work.)
"""

from models_ml.classifier import SUPPORTED_LANGUAGES

LANGUAGE_NAMES = SUPPORTED_LANGUAGES

# Human-friendly phrasing per canonical symptom key, used to build the
# "patient appears to be reporting..." sentence.
SYMPTOM_PHRASES = {
    "chest_pain": "chest pain",
    "breathlessness": "difficulty breathing / shortness of breath",
    "severe_bleeding": "heavy/uncontrolled bleeding",
    "unconscious": "loss of consciousness or unresponsiveness",
    "stroke_signs": "possible stroke symptoms (facial drooping, slurred speech, or one-sided weakness)",
    "high_fever": "a high fever",
    "fever": "fever",
    "abdominal_pain": "abdominal/stomach pain",
    "vomiting": "vomiting or nausea",
    "headache": "headache",
    "cough_cold": "cough, cold, or sore throat",
    "fracture_injury": "a possible fracture or severe injury",
    "minor_pain": "general body/joint/back pain",
    "skin_rash": "a skin rash, itching, or allergic reaction",
    "pregnancy_labor": "labor pain or a pregnancy-related concern",
    "chest_infection": "chest congestion or wheezing",
    "mental_distress": "anxiety, panic, or emotional distress",
}

URGENCY_RATIONALE = {
    "Emergency": (
        "This severity level indicates symptoms that can be life-threatening "
        "within minutes to hours and typically require immediate physician "
        "evaluation, stabilization, and emergency department resources."
    ),
    "Urgent": (
        "This severity level indicates symptoms that need prompt clinical "
        "evaluation, generally within the same day, but do not appear "
        "immediately life-threatening based on the information provided."
    ),
    "Non-Urgent": (
        "This severity level indicates symptoms that can typically be "
        "evaluated through a routine outpatient visit."
    ),
}


def generate_summary(raw_text, language, urgency, score, department, matched_terms, api_key=None):
    """
    Builds a structured, rule-based clinical triage note.

    `api_key` is accepted for backward compatibility but is not used —
    this version never makes any external/API call.
    """
    lang_name = LANGUAGE_NAMES.get(language, "English")

    if matched_terms:
        phrases = [SYMPTOM_PHRASES.get(t, t.replace("_", " ")) for t in dict.fromkeys(matched_terms)]
        if len(phrases) == 1:
            symptom_sentence = f"The patient appears to be reporting {phrases[0]}."
        else:
            symptom_sentence = (
                "The patient appears to be reporting " + ", ".join(phrases[:-1]) +
                f", and {phrases[-1]}."
            )
    else:
        symptom_sentence = (
            "No specific high-signal symptom keywords were detected in the "
            "patient's description; a general clinical review is recommended."
        )

    rationale = URGENCY_RATIONALE.get(urgency, "")

    summary = (
        f"Patient submitted this description in {lang_name}: \"{raw_text}\". "
        f"{symptom_sentence} "
        f"Automated triage assessment: {urgency} (severity score {score}/10). "
        f"{rationale} "
        f"Recommended initial routing: {department}. "
        f"This is an automated preliminary summary only — clinical staff must "
        f"independently assess the patient before making any treatment decision."
    )

    return summary, "template"
