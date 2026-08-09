import os
import json
from datetime import datetime

from flask import Flask, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================================================
# APP SETUP
# =========================================================

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# GEMINI SETUP
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please add GEMINI_API_KEY to your .env file."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# RECENT SCANS
# =========================================================

RECENT_SCANS = []


# =========================================================
# FALLBACK RESULT
# =========================================================

def unclear_result():
    return {
        "disease_ta": "தெளிவான முடிவு கிடைக்கவில்லை",
        "disease_en": "Unable to confidently identify the disease",
        "disease_hi": "रोग की स्पष्ट पहचान नहीं हो सकी",

        "remedy_ta": "தெளிவான இலைப் படத்தை மீண்டும் upload செய்யவும்.",
        "remedy_en": "Please upload a clear image of a single leaf for better analysis.",
        "remedy_hi": "बेहतर विश्लेषण के लिए एक साफ पत्ती की तस्वीर अपलोड करें.",

        "prevention_ta": "நல்ல வெளிச்சத்தில், இலை தெளிவாகத் தெரியும் வகையில் படம் எடுக்கவும்.",
        "prevention_en": "Take a clear photo in good lighting with the leaf clearly visible.",
        "prevention_hi": "अच्छी रोशनी में पत्ती को साफ दिखाई देने वाली तस्वीर लें.",

        "suggestions_ta": [
            "ஒரே ஒரு இலை தெளிவாகத் தெரியும் வகையில் படம் எடுக்கவும்",
            "Blur இல்லாமல் படம் எடுக்கவும்",
            "இலைக்கு போதுமான வெளிச்சம் இருக்க வேண்டும்",
        ],

        "suggestions_en": [
            "Capture one leaf clearly",
            "Avoid blurry images",
            "Use sufficient lighting",
        ],

        "suggestions_hi": [
            "एक पत्ती को साफ दिखाई देने वाली तस्वीर लें",
            "धुंधली तस्वीर से बचें",
            "पर्याप्त रोशनी रखें",
        ],

        "confidence": 0,
        "crop": "Unknown",
        "symptoms": "Image quality is insufficient for reliable analysis.",
        "cause": "Unknown",
    }


# =========================================================
# SELECT UPLOADED FILE
# =========================================================

def selected_upload():
    for field in ("camera_image", "gallery_image", "image"):
        file = request.files.get(field)

        if file and file.filename:
            return file

    return None


# =========================================================
# GEMINI IMAGE ANALYSIS
# =========================================================

def analyze_leaf_image(filepath):

    try:
        # -------------------------------------------------
        # Validate image
        # -------------------------------------------------

        image = Image.open(filepath).convert("RGB")

        # -------------------------------------------------
        # Convert image to bytes
        # -------------------------------------------------

        import io

        image_bytes = io.BytesIO()

        image.save(image_bytes, format="JPEG")

        image_data = image_bytes.getvalue()

        # -------------------------------------------------
        # Gemini image part
        # -------------------------------------------------

        image_part = types.Part.from_bytes(
            data=image_data,
            mime_type="image/jpeg",
        )

        # -------------------------------------------------
        # Prompt
        # -------------------------------------------------

        prompt = """
You are an agricultural plant disease analysis assistant.

Analyze the uploaded plant leaf image carefully.

Your task is to identify the crop and possible disease based ONLY on
visible evidence in the image.

IMPORTANT RULES:

1. Do not claim 100% certainty.
2. If the image is blurry, unclear, too dark, or does not show a leaf
   clearly, mark the result as uncertain.
3. Do not invent symptoms that are not visible.
4. If you cannot reliably identify the crop or disease, return:
   "Uncertain".
5. Give a confidence score from 0 to 100.
6. The confidence score is an estimate, not a scientifically validated
   probability.
7. Give practical general agricultural advice.
8. Do not recommend dangerous chemical use or unsafe pesticide handling.
9. For severe disease cases, recommend consulting a qualified
   agriculture professional.
10. Return ONLY valid JSON.

Return this exact JSON structure:

{
  "crop": "Crop name",
  "disease_en": "Disease name in English",
  "disease_ta": "Disease name in Tamil",
  "disease_hi": "Disease name in Hindi",

  "confidence": 0,

  "symptoms_en": "Visible symptoms in English",
  "symptoms_ta": "Visible symptoms in Tamil",
  "symptoms_hi": "Visible symptoms in Hindi",

  "cause_en": "Possible cause in English",
  "cause_ta": "Possible cause in Tamil",
  "cause_hi": "Possible cause in Hindi",

  "remedy_en": "General treatment or management advice in English",
  "remedy_ta": "General treatment or management advice in Tamil",
  "remedy_hi": "General treatment or management advice in Hindi",

  "prevention_en": "Prevention advice in English",
  "prevention_ta": "Prevention advice in Tamil",
  "prevention_hi": "Prevention advice in Hindi",

  "suggestions_en": [
    "Suggestion 1",
    "Suggestion 2",
    "Suggestion 3"
  ],

  "suggestions_ta": [
    "Suggestion 1",
    "Suggestion 2",
    "Suggestion 3"
  ],

  "suggestions_hi": [
    "Suggestion 1",
    "Suggestion 2",
    "Suggestion 3"
  ]
}
"""

        # -------------------------------------------------
        # Gemini request
        # -------------------------------------------------

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[
                prompt,
                image_part,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        # -------------------------------------------------
        # Read response
        # -------------------------------------------------

        raw_text = response.text.strip()

        result = json.loads(raw_text)

        # -------------------------------------------------
        # Ensure required fields exist
        # -------------------------------------------------

        result.setdefault("crop", "Unknown")
        result.setdefault("disease_en", "Uncertain")
        result.setdefault("disease_ta", "தெளிவான முடிவு கிடைக்கவில்லை")
        result.setdefault("disease_hi", "रोग की स्पष्ट पहचान नहीं हो सकी")

        result.setdefault("confidence", 0)

        result.setdefault(
            "symptoms_en",
            "No reliable symptoms could be determined.",
        )

        result.setdefault(
            "symptoms_ta",
            "தெளிவான அறிகுறிகளை கண்டறிய முடியவில்லை.",
        )

        result.setdefault(
            "symptoms_hi",
            "स्पष्ट लक्षण निर्धारित नहीं किए जा सके.",
        )

        result.setdefault(
            "cause_en",
            "Unknown",
        )

        result.setdefault(
            "cause_ta",
            "காரணம் தெளிவாக தெரியவில்லை.",
        )

        result.setdefault(
            "cause_hi",
            "कारण स्पष्ट नहीं है.",
        )

        result.setdefault(
            "remedy_en",
            "Consult an agriculture professional for confirmation.",
        )

        result.setdefault(
            "remedy_ta",
            "உறுதிப்படுத்த வேளாண் நிபுணரை அணுகவும்.",
        )

        result.setdefault(
            "remedy_hi",
            "पुष्टि के लिए कृषि विशेषज्ञ से सलाह लें.",
        )

        result.setdefault(
            "prevention_en",
            "Maintain good plant hygiene and monitor the plant regularly.",
        )

        result.setdefault(
            "prevention_ta",
            "செடியை தொடர்ந்து கண்காணித்து நல்ல பராமரிப்பை செய்யவும்.",
        )

        result.setdefault(
            "prevention_hi",
            "पौधे की नियमित निगरानी और अच्छी देखभाल करें.",
        )

        result.setdefault(
            "suggestions_en",
            [],
        )

        result.setdefault(
            "suggestions_ta",
            [],
        )

        result.setdefault(
            "suggestions_hi",
            [],
        )

        # -------------------------------------------------
        # Convert confidence to safe integer
        # -------------------------------------------------

        try:
            result["confidence"] = int(result["confidence"])
        except (ValueError, TypeError):
            result["confidence"] = 0

        result["confidence"] = max(
            0,
            min(100, result["confidence"])
        )

        # -------------------------------------------------
        # Add metrics key so existing template won't fail
        # -------------------------------------------------

        result["metrics"] = {
            "ai_analysis": True,
            "model": "gemini-3.5-flash",
        }

        return result

    except (
        UnidentifiedImageError,
        OSError,
    ):

        return unclear_result()

    except json.JSONDecodeError:

        print("Gemini returned invalid JSON.")

        return unclear_result()

    except Exception as e:

        print("Gemini API error:", e)

        result = unclear_result()

        result["disease_en"] = "AI analysis failed"
        result["disease_ta"] = "AI பகுப்பாய்வு தோல்வியடைந்தது"
        result["disease_hi"] = "AI विश्लेषण विफल हुआ"

        return result


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        recent_scans=RECENT_SCANS,
        error=None,
    )


# =========================================================
# SERVE UPLOADED IMAGES
# =========================================================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
    )


# =========================================================
# PREDICT ROUTE
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    lang = request.form.get("lang", "ta")

    file = selected_upload()

    # -----------------------------------------------------
    # No image
    # -----------------------------------------------------

    if not file:

        return render_template(
            "index.html",
            recent_scans=RECENT_SCANS,
            error="Please choose or take one photo first.",
        )

    # -----------------------------------------------------
    # Secure filename
    # -----------------------------------------------------

    filename = secure_filename(file.filename)

    # If filename is empty
    if not filename:

        return render_template(
            "index.html",
            recent_scans=RECENT_SCANS,
            error="Invalid image file.",
        )

    # -----------------------------------------------------
    # Save image
    # -----------------------------------------------------

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename,
    )

    file.save(filepath)

    uploaded_url = f"/uploads/{filename}"

    # -----------------------------------------------------
    # Gemini AI analysis
    # -----------------------------------------------------

    result = analyze_leaf_image(filepath)

    # -----------------------------------------------------
    # Recent scan
    # -----------------------------------------------------

    RECENT_SCANS.insert(
        0,
        {
            "title_ta": result.get(
                "disease_ta",
                "தெளிவான முடிவு இல்லை",
            ),

            "title_en": result.get(
                "disease_en",
                "Uncertain",
            ),

            "title_hi": result.get(
                "disease_hi",
                "अनिश्चित",
            ),

            "date": datetime.now().strftime(
                "%d %b %Y"
            ),

            "image_url": uploaded_url,
        },
    )

    # Keep only latest 5
    del RECENT_SCANS[5:]

    # -----------------------------------------------------
    # Result page
    # -----------------------------------------------------

    return render_template(
        "result.html",
        lang=lang,
        result=result,
        uploaded_url=uploaded_url,
        recent_scans=RECENT_SCANS,
    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )