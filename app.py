import os
import json
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for,
    flash
)

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from dotenv import load_dotenv
from google import genai
from google.genai import types

from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# =========================================================
# APP SETUP
# =========================================================

load_dotenv()

app = Flask(__name__)

# IMPORTANT:
# Add SECRET_KEY in Render Environment Variables.
app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "development-secret-key-change-this"
)


# =========================================================
# DATABASE SETUP
# =========================================================

database_url = os.getenv("DATABASE_URL")

if database_url:

    if database_url.startswith("postgres://"):

        database_url = database_url.replace(
            "postgres://",
            "postgresql://",
            1
        )

    app.config["SQLALCHEMY_DATABASE_URI"] = database_url

else:

    # Local fallback
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        "sqlite:///farmer_ai.db"
    )


app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# =========================================================
# LOGIN MANAGER
# =========================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"

login_manager.login_message = (
    "Please login to access Farmer AI."
)


# =========================================================
# USER MODEL
# =========================================================

class User(UserMixin, db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    role = db.Column(
        db.String(20),
        default="user",
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


# =========================================================
# SCAN MODEL
# =========================================================

class Scan(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    image_url = db.Column(
        db.String(500),
        nullable=False
    )

    crop = db.Column(
        db.String(100),
        default="Unknown"
    )

    disease_en = db.Column(
        db.String(200),
        default="Uncertain"
    )

    disease_ta = db.Column(
        db.String(300),
        default="தெளிவான முடிவு கிடைக்கவில்லை"
    )

    disease_hi = db.Column(
        db.String(300),
        default="रोग की स्पष्ट पहचान नहीं हो सकी"
    )

    confidence = db.Column(
        db.Integer,
        default=0
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


# =========================================================
# USER LOADER
# =========================================================

@login_manager.user_loader
def load_user(user_id):

    return db.session.get(
        User,
        int(user_id)
    )


# =========================================================
# UPLOAD SETUP
# =========================================================

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# GEMINI SETUP
# =========================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please add GEMINI_API_KEY to your environment variables."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# FALLBACK RESULT
# =========================================================

def unclear_result():

    return {

        "disease_ta":
            "தெளிவான முடிவு கிடைக்கவில்லை",

        "disease_en":
            "Unable to confidently identify the disease",

        "disease_hi":
            "रोग की स्पष्ट पहचान नहीं हो सकी",

        "remedy_ta":
            "தெளிவான இலைப் படத்தை மீண்டும் upload செய்யவும்.",

        "remedy_en":
            "Please upload a clear image of a single leaf for better analysis.",

        "remedy_hi":
            "बेहतर विश्लेषण के लिए एक साफ पत्ती की तस्वीर अपलोड करें.",

        "prevention_ta":
            "நல்ல வெளிச்சத்தில், இலை தெளிவாகத் தெரியும் வகையில் படம் எடுக்கவும்.",

        "prevention_en":
            "Take a clear photo in good lighting with the leaf clearly visible.",

        "prevention_hi":
            "अच्छी रोशनी में पत्ती को साफ दिखाई देने वाली तस्वीर लें.",

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

        "symptoms":
            "Image quality is insufficient for reliable analysis.",

        "cause": "Unknown",
    }


# =========================================================
# SELECT UPLOADED FILE
# =========================================================

def selected_upload():

    for field in (
        "camera_image",
        "gallery_image",
        "image"
    ):

        file = request.files.get(field)

        if file and file.filename:

            return file

    return None


# =========================================================
# GEMINI IMAGE ANALYSIS
# =========================================================

def analyze_leaf_image(filepath):

    try:

        image = Image.open(
            filepath
        ).convert("RGB")


        # -------------------------------------------------
        # Convert image to bytes
        # -------------------------------------------------

        import io

        image_bytes = io.BytesIO()

        image.save(
            image_bytes,
            format="JPEG"
        )

        image_data = image_bytes.getvalue()


        # -------------------------------------------------
        # Gemini image part
        # -------------------------------------------------

        image_part = types.Part.from_bytes(

            data=image_data,

            mime_type="image/jpeg"

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
                image_part
            ],

            config=types.GenerateContentConfig(

                response_mime_type="application/json"

            )

        )


        # -------------------------------------------------
        # Parse response
        # -------------------------------------------------

        raw_text = response.text.strip()

        result = json.loads(
            raw_text
        )


        # -------------------------------------------------
        # Defaults
        # -------------------------------------------------

        result.setdefault(
            "crop",
            "Unknown"
        )

        result.setdefault(
            "disease_en",
            "Uncertain"
        )

        result.setdefault(
            "disease_ta",
            "தெளிவான முடிவு கிடைக்கவில்லை"
        )

        result.setdefault(
            "disease_hi",
            "रोग की स्पष्ट पहचान नहीं हो सकी"
        )

        result.setdefault(
            "confidence",
            0
        )

        result.setdefault(
            "symptoms_en",
            "No reliable symptoms could be determined."
        )

        result.setdefault(
            "symptoms_ta",
            "தெளிவான அறிகுறிகளை கண்டறிய முடியவில்லை."
        )

        result.setdefault(
            "symptoms_hi",
            "स्पष्ट लक्षण निर्धारित नहीं किए जा सके."
        )

        result.setdefault(
            "cause_en",
            "Unknown"
        )

        result.setdefault(
            "cause_ta",
            "காரணம் தெளிவாக தெரியவில்லை."
        )

        result.setdefault(
            "cause_hi",
            "कारण स्पष्ट नहीं है."
        )

        result.setdefault(
            "remedy_en",
            "Consult an agriculture professional for confirmation."
        )

        result.setdefault(
            "remedy_ta",
            "உறுதிப்படுத்த வேளாண் நிபுணரை அணுகவும்."
        )

        result.setdefault(
            "remedy_hi",
            "पुष्टि के लिए कृषि विशेषज्ञ से सलाह लें."
        )

        result.setdefault(
            "prevention_en",
            "Maintain good plant hygiene and monitor the plant regularly."
        )

        result.setdefault(
            "prevention_ta",
            "செடியை தொடர்ந்து கண்காணித்து நல்ல பராமரிப்பை செய்யவும்."
        )

        result.setdefault(
            "prevention_hi",
            "पौधे की नियमित निगरानी और अच्छी देखभाल करें."
        )

        result.setdefault(
            "suggestions_en",
            []
        )

        result.setdefault(
            "suggestions_ta",
            []
        )

        result.setdefault(
            "suggestions_hi",
            []
        )


        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        try:

            result["confidence"] = int(
                result["confidence"]
            )

        except (
            ValueError,
            TypeError
        ):

            result["confidence"] = 0


        result["confidence"] = max(

            0,

            min(
                100,
                result["confidence"]
            )

        )


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        result["metrics"] = {

            "ai_analysis": True,

            "model": "gemini-3.5-flash"

        }


        return result


    except (
        UnidentifiedImageError,
        OSError
    ):

        return unclear_result()


    except json.JSONDecodeError:

        print(
            "Gemini returned invalid JSON."
        )

        return unclear_result()


    except Exception as e:

        print(
            "Gemini API error:",
            e
        )

        result = unclear_result()

        result["disease_en"] = (
            "AI analysis failed"
        )

        result["disease_ta"] = (
            "AI பகுப்பாய்வு தோல்வியடைந்தது"
        )

        result["disease_hi"] = (
            "AI विश्लेषण विफल हुआ"
        )

        return result


# =========================================================
# LOGIN PAGE
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if current_user.is_authenticated:

        return redirect(
            url_for("home")
        )


    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        user = User.query.filter_by(
            email=email
        ).first()


        if user and check_password_hash(
            user.password,
            password
        ):

            login_user(user)

            flash(
                "Login successful!",
                "success"
            )

            if user.role == "admin":

                return redirect(
                    url_for("admin_dashboard")
                )

            return redirect(
                url_for("home")
            )


        flash(
            "Invalid email or password.",
            "error"
        )


    return render_template(
        "login.html"
    )


# =========================================================
# REGISTER PAGE
# =========================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if current_user.is_authenticated:

        return redirect(
            url_for("home")
        )


    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        if not name or not email or not password:

            flash(
                "All fields are required.",
                "error"
            )

            return render_template(
                "register.html"
            )


        existing_user = User.query.filter_by(
            email=email
        ).first()


        if existing_user:

            flash(
                "Email already registered.",
                "error"
            )

            return render_template(
                "register.html"
            )


        new_user = User(

            name=name,

            email=email,

            password=generate_password_hash(
                password
            ),

            role="user"

        )


        db.session.add(
            new_user
        )

        db.session.commit()


        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )


    return render_template(
        "register.html"
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
@login_required
def home():

    user_scans = Scan.query.filter_by(

        user_id=current_user.id

    ).order_by(

        Scan.created_at.desc()

    ).limit(5).all()


    recent_scans = []


    for scan in user_scans:

        recent_scans.append({

            "title_ta": scan.disease_ta,

            "title_en": scan.disease_en,

            "title_hi": scan.disease_hi,

            "date": scan.created_at.strftime(
                "%d %b %Y"
            ),

            "image_url": scan.image_url

        })


    return render_template(

        "index.html",

        recent_scans=recent_scans,

        error=None,

        user=current_user

    )


# =========================================================
# SERVE UPLOADED IMAGES
# =========================================================

@app.route(
    "/uploads/<path:filename>"
)
@login_required
def uploaded_file(filename):

    return send_from_directory(

        app.config["UPLOAD_FOLDER"],

        filename

    )


# =========================================================
# PREDICT
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
def predict():

    lang = request.form.get(
        "lang",
        "ta"
    )

    file = selected_upload()


    if not file:

        return render_template(

            "index.html",

            recent_scans=[],

            error="Please choose or take one photo first.",

            user=current_user

        )


    filename = secure_filename(
        file.filename
    )


    if not filename:

        return render_template(

            "index.html",

            recent_scans=[],

            error="Invalid image file.",

            user=current_user

        )


    # -----------------------------------------------------
    # Avoid duplicate filenames
    # -----------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d%H%M%S"
    )

    filename = (
        f"{current_user.id}_{timestamp}_{filename}"
    )


    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )


    file.save(
        filepath
    )


    uploaded_url = (
        f"/uploads/{filename}"
    )


    # -----------------------------------------------------
    # AI prediction
    # -----------------------------------------------------

    result = analyze_leaf_image(
        filepath
    )


    # -----------------------------------------------------
    # SAVE SCAN TO DATABASE
    # -----------------------------------------------------

    scan = Scan(

        user_id=current_user.id,

        image_url=uploaded_url,

        crop=result.get(
            "crop",
            "Unknown"
        ),

        disease_en=result.get(
            "disease_en",
            "Uncertain"
        ),

        disease_ta=result.get(
            "disease_ta",
            "தெளிவான முடிவு கிடைக்கவில்லை"
        ),

        disease_hi=result.get(
            "disease_hi",
            "रोग की स्पष्ट पहचान नहीं हो सकी"
        ),

        confidence=result.get(
            "confidence",
            0
        )

    )


    db.session.add(
        scan
    )

    db.session.commit()


    # -----------------------------------------------------
    # Recent scans
    # -----------------------------------------------------

    user_scans = Scan.query.filter_by(

        user_id=current_user.id

    ).order_by(

        Scan.created_at.desc()

    ).limit(5).all()


    recent_scans = []


    for item in user_scans:

        recent_scans.append({

            "title_ta": item.disease_ta,

            "title_en": item.disease_en,

            "title_hi": item.disease_hi,

            "date": item.created_at.strftime(
                "%d %b %Y"
            ),

            "image_url": item.image_url

        })


    # -----------------------------------------------------
    # Result page
    # -----------------------------------------------------

    return render_template(

        "result.html",

        lang=lang,

        result=result,

        uploaded_url=uploaded_url,

        recent_scans=recent_scans,

        user=current_user

    )


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/admin")
@login_required
def admin_dashboard():

    if current_user.role != "admin":

        flash(
            "Admin access required.",
            "error"
        )

        return redirect(
            url_for("home")
        )


    users = User.query.order_by(
        User.created_at.desc()
    ).all()


    scans = Scan.query.order_by(
        Scan.created_at.desc()
    ).all()


    return render_template(

        "admin.html",

        users=users,

        scans=scans,

        user=current_user

    )


# =========================================================
# DATABASE TABLE CREATION
# =========================================================

with app.app_context():

    db.create_all()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )