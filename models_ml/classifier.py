"""
Multilingual symptom -> urgency/department classifier.

DESIGN NOTE (read this):
This module ships a fast, dependency-free, rule/lexicon-based classifier that
works OFFLINE and gives the app a fully working baseline today, across
English, Hindi, Marathi, Tamil and Telugu, without needing to download or
host a large model.

It is deliberately structured so you can swap the `score_symptoms()` function
for a real fine-tuned multilingual transformer (e.g. MuRIL, IndicBERT,
mBERT, or an Indic-tuned LLaMA/Gemma checkpoint) with almost no change to
the rest of the app:

    1. Fine-tune / host a classification model that outputs
       (urgency_label, confidence, department_label).
    2. Replace the body of `score_symptoms()` with a call to that model
       (local `transformers` pipeline, or a hosted inference endpoint).
    3. Keep the same return shape: {"urgency": ..., "score": ..., "department": ...,
       "matched_terms": [...]}

Everything downstream (Flask routes, DB, templates, LLM summary step) is
already built against that contract.
"""

import re

# ---------------------------------------------------------------------------
# Symptom lexicon: English canonical key -> translations in each language.
# Extend this freely; it's the "training data" of the rule-based model.
# ---------------------------------------------------------------------------

SYMPTOM_LEXICON = {
    # canonical_key: {lang: [surface forms...]}
    "chest_pain": {
        "en": ["chest pain", "pain in chest", "tightness in chest"],
        "hi": ["सीने में दर्द", "छाती में दर्द", "सीने में जकड़न"],
        "hg": ["chest me dard", "seene me dard", "chhati me dard", "chest pain ho raha", "seene me jakdan"],
        "mr": ["छातीत दुखणे", "छातीत वेदना"],
        "ta": ["மார்பு வலி", "நெஞ்சு வலி"],
        "te": ["ఛాతీ నొప్పి", "గుండె నొప్పి"],
    },
    "breathlessness": {
        "en": ["shortness of breath", "difficulty breathing", "cant breathe", "breathless"],
        "hi": ["सांस लेने में तकलीफ", "सांस फूलना", "सांस की तकलीफ"],
        "hg": ["saans lene me takleef", "saans phool rahi", "saans lene me problem", "saans nahi aa rahi"],
        "mr": ["श्वास घ्यायला त्रास", "धाप लागणे"],
        "ta": ["மூச்சு திணறல்", "மூச்சு வாங்குவதில் சிரமம்"],
        "te": ["ఊపిరి ఆడకపోవడం", "శ్వాస తీసుకోవడంలో ఇబ్బంది"],
    },
    "severe_bleeding": {
        "en": ["heavy bleeding", "severe bleeding", "uncontrolled bleeding"],
        "hi": ["तेज़ खून बहना", "अत्यधिक रक्तस्राव"],
        "hg": ["zyada khoon beh raha", "bleeding nahi ruk rahi", "bahut khoon nikal raha"],
        "mr": ["जास्त रक्तस्त्राव"],
        "ta": ["அதிக இரத்தப்போக்கு"],
        "te": ["ఎక్కువ రక్తస్రావం"],
    },
    "unconscious": {
        "en": ["unconscious", "not responding", "fainted", "passed out"],
        "hi": ["बेहोश", "होश नहीं आ रहा"],
        "hg": ["behosh ho gaya", "hosh nahi aa raha", "behosh ho gayi"],
        "mr": ["बेशुद्ध"],
        "ta": ["மயக்கம்", "உணர்வு இழந்தார்"],
        "te": ["అపస్మారక స్థితి", "స్పృహ కోల్పోవడం"],
    },
    "stroke_signs": {
        "en": ["face drooping", "slurred speech", "one side weakness", "sudden numbness"],
        "hi": ["चेहरा टेढ़ा होना", "बोलने में लड़खड़ाहट", "एक तरफ कमजोरी"],
        "hg": ["chehra tedha ho gaya", "bolne me ladkhadahat", "ek taraf kamzori", "muh tedha ho gaya"],
        "mr": ["चेहरा वाकडा होणे", "बोलण्यात अडखळणे"],
        "ta": ["முகம் கோணுதல்", "பேச்சு தடுமாற்றம்"],
        "te": ["ముఖం వంకరపోవడం", "మాటలు తడబడటం"],
    },
    "high_fever": {
        "en": ["high fever", "very high temperature"],
        "hi": ["तेज़ बुखार", "बहुत तेज़ बुखार"],
        "hg": ["tez bukhar", "bahut tez bukhar hai", "bukhar bahut zyada hai"],
        "mr": ["जास्त ताप"],
        "ta": ["அதிக காய்ச்சல்"],
        "te": ["అధిక జ్వరం"],
    },
    "fever": {
        "en": ["fever", "temperature", "feeling feverish"],
        "hi": ["बुखार"],
        "hg": ["bukhar", "bukhar hai", "fever aa raha hai"],
        "mr": ["ताप"],
        "ta": ["காய்ச்சல்"],
        "te": ["జ్వరం"],
    },
    "abdominal_pain": {
        "en": ["stomach pain", "abdominal pain", "belly pain"],
        "hi": ["पेट में दर्द"],
        "hg": ["pet me dard", "pet dard kar raha hai", "stomach me dard"],
        "mr": ["पोटदुखी", "पोटात दुखणे"],
        "ta": ["வயிற்று வலி"],
        "te": ["కడుపు నొప్పి"],
    },
    "vomiting": {
        "en": ["vomiting", "throwing up", "nausea"],
        "hi": ["उल्टी", "जी मिचलाना"],
        "hg": ["ulti ho rahi hai", "jee michla raha hai", "vomit ho raha"],
        "mr": ["उलटी", "मळमळणे"],
        "ta": ["வாந்தி", "குமட்டல்"],
        "te": ["వాంతులు", "వికారం"],
    },
    "headache": {
        "en": ["headache", "head pain"],
        "hi": ["सिरदर्द"],
        "hg": ["sar dard", "sir me dard", "headache ho raha hai"],
        "mr": ["डोकेदुखी"],
        "ta": ["தலைவலி"],
        "te": ["తలనొప్పి"],
    },
    "cough_cold": {
        "en": ["cough", "cold", "runny nose", "sore throat"],
        "hi": ["खांसी", "जुकाम", "गले में खराश"],
        "hg": ["khansi ho rahi hai", "jukam hai", "gale me kharash", "cold hai"],
        "mr": ["खोकला", "सर्दी", "घसा खवखवणे"],
        "ta": ["இருமல்", "சளி", "தொண்டை புண்"],
        "te": ["దగ్గు", "జలుబు", "గొంతు నొప్పి"],
    },
    "fracture_injury": {
        "en": ["fracture", "broken bone", "severe injury", "accident injury"],
        "hi": ["हड्डी टूटना", "फ्रैक्चर", "गंभीर चोट"],
        "hg": ["haddi toot gayi", "fracture ho gaya", "accident me chot lagi", "badi chot lagi hai"],
        "mr": ["हाड मोडणे", "गंभीर दुखापत"],
        "ta": ["எலும்பு முறிவு", "விபத்து காயம்"],
        "te": ["ఎముక విరగడం", "తీవ్రమైన గాయం"],
    },
    "minor_pain": {
        "en": ["body pain", "mild pain", "back pain", "joint pain"],
        "hi": ["शरीर में दर्द", "पीठ दर्द", "जोड़ों का दर्द"],
        "hg": ["body pain ho raha hai", "peeth me dard", "joint pain hai", "sharir me dard"],
        "mr": ["अंगदुखी", "पाठदुखी", "सांधेदुखी"],
        "ta": ["உடல் வலி", "முதுகு வலி", "மூட்டு வலி"],
        "te": ["ఒంటి నొప్పులు", "వెన్ను నొప్పి", "కీళ్ల నొప్పులు"],
    },
    "skin_rash": {
        "en": ["skin rash", "itching", "allergy"],
        "hi": ["त्वचा पर चकत्ते", "खुजली", "एलर्जी"],
        "hg": ["skin par rash hai", "khujli ho rahi hai", "allergy ho gayi"],
        "mr": ["त्वचेवर पुरळ", "खाज", "ऍलर्जी"],
        "ta": ["தோல் அரிப்பு", "ஒவ்வாமை"],
        "te": ["చర్మం దద్దుర్లు", "దురద", "అలర్జీ"],
    },
    "pregnancy_labor": {
        "en": ["labor pain", "pregnant", "contractions"],
        "hi": ["प्रसव पीड़ा", "गर्भवती"],
        "hg": ["labor pain ho raha hai", "pregnant hu", "delivery pain shuru hua"],
        "mr": ["प्रसूती वेदना", "गरोदर"],
        "ta": ["பிரசவ வலி", "கர்ப்பிணி"],
        "te": ["ప్రసవ నొప్పులు", "గర్భవతి"],
    },
    "chest_infection": {
        "en": ["wheezing", "chest congestion"],
        "hi": ["सांस में घरघराहट", "छाती में जकड़न"],
        "hg": ["saans me ghar ghar awaaz aati hai", "chest me jakdan hai"],
        "mr": ["श्वासात घरघर", "छातीत जड जाणवणे"],
        "ta": ["மூச்சில் சத்தம்", "நெஞ்சு பிடிப்பு"],
        "te": ["ఊపిరిలో గురక", "ఛాతీ బిగుసుకుపోవడం"],
    },
    "mental_distress": {
        "en": ["anxiety", "panic attack", "cant sleep", "depressed", "suicidal thoughts"],
        "hi": ["चिंता", "घबराहट", "नींद नहीं आना", "उदासी", "आत्महत्या के विचार"],
        "hg": ["bahut tension ho rahi hai", "ghabrahat ho rahi hai", "neend nahi aa rahi", "man udaas hai"],
        "mr": ["चिंता", "घबराट", "झोप न येणे", "नैराश्य"],
        "ta": ["பதற்றம்", "தூக்கமின்மை", "மனச்சோர்வு"],
        "te": ["ఆందోళన", "నిద్ర పట్టకపోవడం", "నిరాశ"],
    },
}

# Department each canonical symptom key maps to
DEPARTMENT_MAP = {
    "chest_pain": "Cardiology / Emergency",
    "breathlessness": "Pulmonology / Emergency",
    "severe_bleeding": "Emergency / Trauma",
    "unconscious": "Emergency",
    "stroke_signs": "Neurology / Emergency",
    "high_fever": "General Medicine",
    "fever": "General Medicine",
    "abdominal_pain": "Gastroenterology",
    "vomiting": "General Medicine",
    "headache": "Neurology (OPD)",
    "cough_cold": "General Medicine / ENT",
    "fracture_injury": "Orthopedics / Emergency",
    "minor_pain": "Orthopedics (OPD)",
    "skin_rash": "Dermatology",
    "pregnancy_labor": "Obstetrics & Gynecology",
    "chest_infection": "Pulmonology",
    "mental_distress": "Psychiatry",
}

# Urgency weight per canonical symptom key (0-10 scale, 10 = most critical)
URGENCY_WEIGHT = {
    "chest_pain": 10,
    "breathlessness": 9,
    "severe_bleeding": 10,
    "unconscious": 10,
    "stroke_signs": 10,
    "fracture_injury": 7,
    "high_fever": 6,
    "pregnancy_labor": 8,
    "abdominal_pain": 5,
    "chest_infection": 6,
    "vomiting": 4,
    "mental_distress": 6,
    "headache": 3,
    "fever": 3,
    "cough_cold": 2,
    "minor_pain": 2,
    "skin_rash": 2,
}

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "hg": "Hinglish",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
}


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def score_symptoms(text: str, language: str):
    """
    Core classification function. Returns a dict:
    {
      "urgency": "Emergency" | "Urgent" | "Non-Urgent",
      "score": int (1-10),
      "department": str,
      "matched_terms": [canonical_key, ...]
    }

    Replace this function's body with a real model call to upgrade from
    lexicon-matching to a fine-tuned multilingual transformer.
    """
    language = language if language in SUPPORTED_LANGUAGES else "en"
    normalized = _normalize(text)

    matched_keys = []
    for canonical_key, translations in SYMPTOM_LEXICON.items():
        surface_forms = translations.get(language, []) + translations.get("en", [])
        for phrase in surface_forms:
            if _normalize(phrase) in normalized:
                matched_keys.append(canonical_key)
                break

    if not matched_keys:
        return {
            "urgency": "Non-Urgent",
            "score": 1,
            "department": "General Medicine (OPD)",
            "matched_terms": [],
        }

    max_weight = max(URGENCY_WEIGHT.get(k, 2) for k in matched_keys)

    if max_weight >= 8:
        urgency = "Emergency"
    elif max_weight >= 5:
        urgency = "Urgent"
    else:
        urgency = "Non-Urgent"

    # department = department of the highest-severity matched symptom
    top_key = max(matched_keys, key=lambda k: URGENCY_WEIGHT.get(k, 2))
    department = DEPARTMENT_MAP.get(top_key, "General Medicine (OPD)")

    return {
        "urgency": urgency,
        "score": max_weight,
        "department": department,
        "matched_terms": matched_keys,
    }
