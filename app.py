from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import os
import time
import json
import torch
import torch.nn as nn

from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)

CORS(app, supports_credentials=True)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config["SECRET_KEY"] = "agro_sage_secret"

app.config["SQLALCHEMY_DATABASE_URI"] = \
    "sqlite:///" + os.path.join(BASE_DIR, "agro_sage.db")

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db = SQLAlchemy(app)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\n🚀 DEVICE: {device}")

MODEL_PATH = os.path.join(BASE_DIR, "agri_model.pth")

CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")

SOLUTION_PATH = os.path.join(
    BASE_DIR,
    "disease_solution.json"
)

model = None

class_names = {}

transform = transforms.Compose([

    transforms.Resize((300, 300)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

if os.path.exists(SOLUTION_PATH):

    with open(SOLUTION_PATH, "r") as f:
        DISEASE_DB = json.load(f)

    print("✅ disease_solution.json loaded")

else:

    DISEASE_DB = {}

    print("⚠️ disease_solution.json missing")

def load_ai_model():

    global model, class_names

    try:

        if not os.path.exists(CLASSES_PATH):

            print("❌ classes.json missing")

            return

        with open(CLASSES_PATH, "r") as f:

            class_names = json.load(f)

        if isinstance(class_names, list):

            class_names = {
                i: name for i, name in enumerate(class_names)
            }

        elif isinstance(class_names, dict):

            class_names = {
                int(k): v
                for k, v in class_names.items()
            }

        else:
            raise ValueError("Invalid classes.json")

        num_classes = len(class_names)

        print(f"✅ Classes Loaded: {num_classes}")

        if not os.path.exists(MODEL_PATH):

            print("❌ best_model.pth missing")

            return

        class AgriCNN(nn.Module):

            def __init__(self, num_classes):

                super().__init__()

                self.backbone = models.efficientnet_b3(
                    weights=None
                )

                in_features = \
                    self.backbone.classifier[1].in_features

                self.backbone.classifier = nn.Identity()

                self.classifier = nn.Sequential(

                    nn.Linear(in_features, 512),

                    nn.ReLU(),

                    nn.Dropout(0.4),

                    nn.Linear(512, num_classes)
                )

            def forward(self, x):

                x = self.backbone(x)

                x = self.classifier(x)

                return x

        model = AgriCNN(num_classes)

        state_dict = torch.load(
            MODEL_PATH,
            map_location=device
        )

        model.load_state_dict(state_dict)

        model.to(device)

        model.eval()

        print("✅ AI MODEL LOADED SUCCESSFULLY")

    except Exception as e:

        print("❌ MODEL LOAD ERROR:")
        print(e)

        model = None

class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100)
    )

    email = db.Column(
        db.String(120),
        unique=True
    )

    password = db.Column(
        db.String(200)
    )

    created_at = db.Column(
        db.Float,
        default=time.time
    )

class ScanHistory(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer
    )

    diagnosis = db.Column(
        db.String(200)
    )

    confidence = db.Column(
        db.String(20)
    )

    image_path = db.Column(
        db.String(300)
    )

    timestamp = db.Column(
        db.Float,
        default=time.time
    )

@app.route("/register", methods=["POST"])
def register():

    try:

        data = request.json

        name = data.get("name")

        email = data.get("email", "").lower()

        password = data.get("password")

        if not name or not email or not password:

            return jsonify({
                "error": "Missing fields"
            }), 400

        existing = User.query.filter_by(
            email=email
        ).first()

        if existing:

            return jsonify({
                "error": "Email already exists"
            }), 409

        new_user = User(

            name=name,

            email=email,

            password=generate_password_hash(password)
        )

        db.session.add(new_user)

        db.session.commit()

        return jsonify({
            "message": "Registered successfully"
        })

    except Exception as e:

        print("REGISTER ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500

@app.route("/login", methods=["POST"])
def login():

    try:

        data = request.json

        email = data.get("email", "").lower()

        password = data.get("password")

        user = User.query.filter_by(
            email=email
        ).first()

        if not user:

            return jsonify({
                "error": "User not found"
            }), 404

        if not check_password_hash(
            user.password,
            password
        ):

            return jsonify({
                "error": "Wrong password"
            }), 401

        return jsonify({

            "message": "Login successful",

            "user": {

                "id": user.id,

                "name": user.name,

                "email": user.email
            }
        })

    except Exception as e:

        print("LOGIN ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}

def allowed_file(filename):

    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() \
        in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if model is None:

            return jsonify({
                "error": "AI model not loaded"
            }), 500

        if "file" not in request.files:

            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "error": "No selected file"
            }), 400

        if not allowed_file(file.filename):

            return jsonify({
                "error": "Invalid file type"
            }), 400

        filename = secure_filename(

            f"{int(time.time())}_{file.filename}"
        )

        path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(path)

        image = Image.open(path).convert("RGB")

        img_tensor = transform(image)

        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():

            outputs = model(img_tensor)

            probs = torch.softmax(
                outputs[0],
                dim=0
            )

            confidence, pred = torch.max(
                probs,
                0
            )

        prediction = class_names[pred.item()]

        confidence_score = \
            f"{confidence.item()*100:.2f}%"

        print("\n🌾 Prediction:", prediction)

        print("🔥 Confidence:", confidence_score)

        disease_key = (
            prediction
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        print("Prediction:", prediction)
        print("Disease Key:", disease_key)


        import re
        from difflib import get_close_matches

        disease_key = re.sub(r"[^a-z0-9]+", "_", prediction.lower().strip())

        print("🔍 Searching key:", disease_key)

        solution = DISEASE_DB.get(disease_key)

        if not solution:

            match = get_close_matches(disease_key, DISEASE_DB.keys(), n=1, cutoff=0.6)

            if match:
                print("⚡ Fuzzy matched:", match[0])
                solution = DISEASE_DB.get(match[0])

        if not solution:
            print("❌ No match found in DB")

            solution = {
                "fertilizers": ["No data available"],
                "chemicals": ["No data available"],
                "organic": ["No data available"],
                "prevention": ["No data available"]
            }

        history = ScanHistory(

            user_id=1,

            diagnosis=prediction,

            confidence=confidence_score,

            image_path=path
        )

        db.session.add(history)

        db.session.commit()

        return jsonify({

            "prediction":
                prediction.replace("_", " "),

            "confidence":
                confidence_score,

            "fertilizers":
                solution.get(
                    "fertilizers",
                    []
                ),

            "chemicals":
                solution.get(
                    "chemicals",
                    []
                ),

            "organic":
                solution.get(
                    "organic",
                    []
                ),

            "prevention":
                solution.get(
                    "prevention",
                    []
                )
        })

    except Exception as e:

        print("❌ PREDICTION ERROR:")
        print(e)

        return jsonify({
            "error": str(e)
        }), 500

@app.route("/history/<int:user_id>")
def history(user_id):

    scans = ScanHistory.query.filter_by(
        user_id=user_id
    ).all()

    return jsonify([

        {
            "diagnosis": s.diagnosis,

            "confidence": s.confidence,

            "image": s.image_path
        }

        for s in scans
    ])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/marketplace")
def marketplace():
    return render_template("market.html")



@app.route("/suggest-page")
def suggest_page():

    return render_template("suggest.html")



import re
from difflib import get_close_matches

def normalize(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower().strip())

@app.route("/suggest", methods=["POST"])
def suggest_api():
    try:
        data = request.get_json()
        disease = data.get("disease", "")

        if not disease:
            return jsonify({"error": "No disease provided"}), 400

        user_key = normalize(disease)

        print("🔍 User input key:", user_key)

        if user_key in DISEASE_DB:
            solution = DISEASE_DB[user_key]
            matched = user_key

        else:
            match = get_close_matches(user_key, DISEASE_DB.keys(), n=1, cutoff=0.3)

            if match:
                matched = match[0]
                solution = DISEASE_DB[matched]
                print("⚡ Fuzzy matched:", matched)

            else:
                matched = None
                for key in DISEASE_DB.keys():
                    if any(word in key for word in user_key.split("_")):
                        matched = key
                        break

                solution = DISEASE_DB.get(matched, {
                    "chemicals": [],
                    "fertilizers": [],
                    "organic": [],
                    "prevention": []
                })

        return jsonify({
            "matched_disease": matched,
            "chemicals": solution.get("chemicals", []),
            "fertilizers": solution.get("fertilizers", []),
            "organic": solution.get("organic", []),
            "prevention": solution.get("prevention", [])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":

    with app.app_context():

        db.create_all()

    load_ai_model()

    app.run(
        use_reloader=False,
        debug=True,
        host="0.0.0.0",
        port=5000
    )
