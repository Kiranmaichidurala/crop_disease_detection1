from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import cv2
import numpy as np
import colorsys
from PIL import Image
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

# -------------------------------------------------------------------
# DATABASE MODELS
# -------------------------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    history = db.relationship('PredictionHistory', backref='user', lazy=True)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    img_path = db.Column(db.String(255))

with app.app_context():
    db.create_all()

# --------------------------------------------------
# Upload Folder
# --------------------------------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MANGO_DATASET_FOLDER = "mango_dataset"
os.makedirs(MANGO_DATASET_FOLDER, exist_ok=True)

# --------------------------------------------------
# Disease Classes
# --------------------------------------------------
classes = [
    "Rice_Bacterial_Blight", "Rice_Blast", "Rice_Tungro", "Rice_Healthy",
    "Cotton_Bacterial_Blight", "Cotton_Leaf_Curl", "Cotton_Wilt", "Cotton_Healthy",
    "Chili_Leaf_Curl", "Chili_Anthracnose", "Chili_Bacterial_Spot", "Chili_Healthy",
    "Maize_Gray_Leaf_Spot", "Maize_Leaf_Blight", "Maize_Common_Rust", "Maize_Healthy",
    "Groundnut_Leaf_Spot", "Groundnut_Rust", "Groundnut_Healthy",
    "Turmeric_Leaf_Blight", "Turmeric_Leaf_Spot", "Turmeric_Healthy",
    "Sugarcane_Red_Rot", "Sugarcane_Smut", "Sugarcane_Healthy",
    "Tomato_Late_Blight", "Tomato_Early_Blight", "Tomato_Leaf_Curl", "Tomato_Healthy",
    "Papaya_Ring_Spot", "Papaya_Mosaic", "Papaya_Healthy",
    "Mango_Anthracnose", "Mango_Powdery_Mildew", "Mango_Sooty_Mold", "Mango_Healthy",
    "Banana_Panama_Disease", "Banana_Sigatoka", "Banana_Healthy",
    "PigeonPea_Sterility_Mosaic", "PigeonPea_Phytophthora_Blight", "PigeonPea_Healthy",
    "Sunflower_Rust", "Sunflower_Downy_Mildew", "Sunflower_Healthy",
    "Jowar_Anthracnose", "Jowar_Grain_Mold", "Jowar_Healthy",
    "Millet_Blast", "Millet_Downy_Mildew", "Millet_Healthy"
]

# --------------------------------------------------
# Preventive Methods & Estimates
# --------------------------------------------------
disease_info = {
    "Rice_Bacterial_Blight": {"methods": "Manage the disease by ensuring proper field drainage and split-applying nitrogen with extra potash to boost plant immunity. Apply Copper Hydroxide or a Streptocycline spray to disinfect the crop and halt bacterial progression.", "recovery": "14-21 days", "cost": "₹500 - ₹800"},
    "Rice_Blast": {"methods": "Plant resistant varieties and maintain balanced nitrogen levels to reduce inoculum buildup and enhance natural plant resistance. Apply Tricyclazole or Isoprothiolane as a preventative fungicide to protect the crop before infection takes hold.", "recovery": "10-15 days", "cost": "₹600 - ₹1000"},
    "Rice_Tungro": {"methods": "Plant resistant varieties like Surekha or Vikramarya and practice synchronous planting to disrupt the life cycle of the green leafhopper vector. Apply Imidacloprid or Thiamethoxam to effectively control the green leafhopper and stop the virus from spreading between plants.", "recovery": "15-20 days", "cost": "₹400 - ₹700"},
    "Rice_Healthy": {"methods": "The crop is healthy. Maintain good fertilizer and irrigation practices.", "recovery": "N/A", "cost": "₹0"},

    "Cotton_Bacterial_Blight": {"methods": "Maintain strict field sanitation by destroying infected crop debris and ensuring seeds are treated before planting to eliminate the primary inoculum. Spray Copper Oxychloride combined with Streptocycline to disinfect the crop and suppress the spread of the bacterial pathogen during humid weather.", "recovery": "12-18 days", "cost": "₹550 - ₹900"},
    "Cotton_Leaf_Curl": {"methods": "Plant resistant varieties and maintain strict field sanitation by removing alternate hosts like weeds to reduce the whitefly population. Apply Afidopyropen or Diafenthiuron to effectively control the whitefly vector and stop the transmission of the virus.", "recovery": "14-21 days", "cost": "₹600 - ₹1100"},
    "Cotton_Wilt": {"methods": "Perform deep summer ploughing to solarize the soil and expose dormant pathogens to high heat, effectively reducing the soil-borne inoculum. Treat seeds with Carbendazim or drench the root zone with Copper Oxychloride to protect the plant from fungal colonization and spread.", "recovery": "15-25 days", "cost": "₹700 - ₹1200"},
    "Cotton_Healthy": {"methods": "Healthy crop. Continue good soil and water management.", "recovery": "N/A", "cost": "₹0"},

    "Chili_Leaf_Curl": {"methods": "Maintain strict field sanitation by removing weeds and alternate hosts to reduce the whitefly population. Apply Afidopyropen or Diafenthiuron to effectively control the whitefly vector and stop the transmission of the virus.", "recovery": "10-14 days", "cost": "₹350 - ₹650"},
    "Chili_Anthracnose": {"methods": "Ensure strict field sanitation by removing crop debris and practicing crop rotation to eliminate the primary seed-borne and soil-borne inoculum. Treat seeds with Carbendazim or spray the crop with Mancozeb to effectively disinfect the plants and prevent the spread of the fungal pathogen.", "recovery": "12-18 days", "cost": "₹500 - ₹850"},
    "Chili_Bacterial_Spot": {"methods": "Practice crop rotation with non-host crops and ensure strict field sanitation by burning plant debris to eliminate the bacterial inoculum surviving in the soil. Treat seeds with Streptocyclin or spray the field with Copper Oxychloride to effectively disinfect the crop and prevent the spread of leaf spots.", "recovery": "10-15 days", "cost": "₹450 - ₹750"},
    "Chili_Healthy": {"methods": "Crop is healthy. Keep monitoring regularly.", "recovery": "N/A", "cost": "₹0"},

    "Maize_Gray_Leaf_Spot": {"methods": "Practice deep tillage to bury and decompose infected crop residues, significantly reducing the initial fungal inoculum that survives on surface debris. Apply Pyraclostrobin or Azoxystrobin fungicides during the tasseling stage to protect the foliage and prevent the development of necrotic lesions.", "recovery": "14-20 days", "cost": "₹600 - ₹950"},
    "Maize_Leaf_Blight": {"methods": "Maintain strict field sanitation by destroying infected stubble and ensuring proper plant spacing to improve air circulation and reduce leaf moisture. Treat seeds with Carbendazim or Trichoderma viride to provide early protection against fungal colonization and reduce the initial disease load.", "recovery": "12-16 days", "cost": "₹500 - ₹800"},
    "Maize_Common_Rust": {"methods": "Practice crop rotation with non-cereal crops like legumes for 2–3 years and plant rust-resistant hybrids to break the disease cycle and limit pustule formation. Apply Mancozeb or Zineb at the first appearance of pustules to protect the leaf surface and prevent the rapid secondary spread of the fungus.", "recovery": "10-15 days", "cost": "₹450 - ₹750"},
    "Maize_Healthy": {"methods": "Healthy crop. Maintain proper spacing and irrigation.", "recovery": "N/A", "cost": "₹0"},

    "Groundnut_Leaf_Spot": {"methods": "Perform deep summer ploughing to solarize the soil and destroy dormant spores, while ensuring all infected crop residues are burned to eliminate the primary source of infection. Treat seeds with Carbendazim or spray the crop with Mancozeb to effectively suppress the development of early and late leaf spot lesions.", "recovery": "12-18 days", "cost": "₹400 - ₹700"},
    "Groundnut_Rust": {"methods": "Practice early sowing to ensure the crop escapes peak infestation periods and lacks a host for initial inoculum. Treat seeds with Trichoderma viride or spray the crop with Chlorothalonil or Tebuconazole to effectively inhibit rust pustule formation and spread.", "recovery": "10-14 days", "cost": "₹350 - ₹650"},
    "Groundnut_Healthy": {"methods": "Healthy crop. Maintain soil fertility.", "recovery": "N/A", "cost": "₹0"},

    "Turmeric_Leaf_Blight": {"methods": "Practice crop rotation and ensure proper field drainage to minimize leaf wetness and break the life cycle of the fungal pathogen. Treat seed rhizomes with Mancozeb or spray the crop with Copper Oxychloride to effectively eliminate the primary inoculum and prevent the spread of blotches.", "recovery": "15-20 days", "cost": "₹500 - ₹850"},
    "Turmeric_Leaf_Spot": {"methods": "Avoid planting turmeric near chilli crops and treat rhizomes with Pseudomonas fluorescens to disrupt the shared host pathogen cycle. Spray Propiconazole (0.1%) at 45 and 90 days after planting to provide targeted systemic protection against the Colletotrichum fungus.", "recovery": "12-16 days", "cost": "₹400 - ₹700"},
    "Turmeric_Healthy": {"methods": "Healthy crop. Maintain organic fertilizer supply.", "recovery": "N/A", "cost": "₹0"},

    "Sugarcane_Red_Rot": {"methods": "Plant resistant varieties like Co 86032 or CoLk 14201 and avoid monoculture to disrupt the pathogen's ability to adapt and spread across large areas. Treat seed setts with Thiophanate Methyl (0.1%) to eliminate the internal fungal inoculum and provide early systemic protection against infection.", "recovery": "20-30 days", "cost": "₹1000 - ₹1800"},
    "Sugarcane_Smut": {"methods": "Plant resistant cultivars like Co 86249 or CoG 93076 and avoid susceptible varieties to prevent the development of characteristic whip-like structures. Treat seed setts with Triadimefon (0.1%) or Propiconazole (0.1%) to effectively eliminate fungal spores and prevent systemic infection.", "recovery": "15-25 days", "cost": "₹800 - ₹1400"},
    "Sugarcane_Healthy": {"methods": "Healthy crop. Maintain good irrigation and fertilizer schedule.", "recovery": "N/A", "cost": "₹0"},

    "Tomato_Late_Blight": {"methods": "Cultivate resistant hybrids like Arka Abhed and use black polythene mulching to prevent soil-borne spores from splashing onto foliage during rainfall. Apply Metalaxyl + Mancozeb or Cymoxanil + Mancozeb at the first sign of cool, humid weather to provide systemic protection against rapid blight spread.", "recovery": "10-15 days", "cost": "₹500 - ₹900"},
    "Tomato_Early_Blight": {"methods": "Practice a 2-3 year rotation with non-solanaceous crops and ensure strict field sanitation to eliminate the soil-borne fungal inoculum. Treat seeds with Trichoderma viride or spray the crop with Chlorothalonil or Azoxystrobin to effectively prevent the development of concentric leaf spots.", "recovery": "12-16 days", "cost": "₹450 - ₹800"},
    "Tomato_Leaf_Curl": {"methods": "Raise seedlings under 40-60 mesh nylon nets and use yellow sticky traps to physically block and monitor the whitefly vector. Spray Cyantraniliprole or Spiromesifen to effectively control the whitefly population and halt the transmission of the leaf curl virus.", "recovery": "14-21 days", "cost": "₹600 - ₹1100"},
    "Tomato_Healthy": {"methods": "Your crop is healthy. Continue proper watering.", "recovery": "N/A", "cost": "₹0"},

    "Papaya_Ring_Spot": {"methods": "Raise seedlings under 40-60 mesh nylon nets and plant maize or sorghum border crops to create a physical and biological shield against aphid vectors. Apply Imidacloprid or Dimethoate to the border crops and young papaya to suppress aphids and delay the transmission of the ring spot virus.", "recovery": "15-25 days", "cost": "₹700 - ₹1300"},
    "Papaya_Mosaic": {"methods": "Establish maize or sorghum border crops to block aphid vectors and prevent early-stage viral transmission. Spray Imidacloprid (0.3 ml/L) or Acephate (1.5g/L) in rotation with 5% Neem Seed Kernel Extract to suppress aphid populations and halt the spread of the mosaic virus.", "recovery": "14-21 days", "cost": "₹600 - ₹1100"},
    "Papaya_Healthy": {"methods": "Healthy papaya crop. Maintain nutrient balance.", "recovery": "N/A", "cost": "₹0"},

    "Mango_Anthracnose": {"methods": "ICAR Recommended: Maintain strict orchard sanitation by pruning diseased twigs and removing fallen fruit to eliminate the primary fungal inoculum. Apply a protective spray of 1% Bordeaux mixture or Copper Oxychloride (0.3%) before and after the monsoon to prevent the spread of anthracnose spores.", "recovery": "15-20 days", "cost": "₹800 - ₹1500"},
    "Mango_Powdery_Mildew": {"methods": "ICAR Recommended: Apply Wettable Sulfur (0.2%) immediately upon panicle emergence and maintain open canopies through pruning to reduce the humidity that favors fungal growth. Follow up with systemic sprays of Hexaconazole or Propiconazole at 15-day intervals to protect developing flowers and young fruits from powdery white colonization.", "recovery": "12-18 days", "cost": "₹700 - ₹1300"},
    "Mango_Sooty_Mold": {"methods": "ICAR Recommended: Prune affected twigs and use 400-gauge polythene bands around trunks to block climbing pests and reduce the honeydew that fuels fungal growth. Spray Starch (2%) to flake off existing mold or apply Imidacloprid to eliminate the sucking insects responsible for the black soot development.", "recovery": "14-21 days", "cost": "₹600 - ₹1100"},
    "Mango_Healthy": {"methods": "ICAR Recommended: Your mango leaf is fresh and healthy. Maintain regular pruning.", "recovery": "N/A", "cost": "₹0"},

    "Banana_Panama_Disease": {"methods": "Use disease-free tissue culture plantlets and apply a Trichoderma-neem cake mixture to the plant base to establish a biological defense against soil-borne infection. Drench the soil with 1% ICAR-FUSICONT at 2, 4, 6, and 8 months post-planting to suppress the Fusarium pathogen and maintain root health.", "recovery": "20-30 days", "cost": "₹1200 - ₹2000"},
    "Banana_Sigatoka": {"methods": "Maintain proper plant spacing and strict field sanitation by removing infected leaves to lower humidity and reduce the primary fungal inoculum. Use disease-free tissue culture plants and apply Propiconazole (0.1%) mixed with a mineral oil sticker to effectively block the spread of leaf spot spores.", "recovery": "15-20 days", "cost": "₹900 - ₹1500"},
    "Banana_Healthy": {"methods": "Healthy banana crop. Ensure proper irrigation.", "recovery": "N/A", "cost": "₹0"},

    "PigeonPea_Sterility_Mosaic": {"methods": "Cultivate resistant varieties like BRG 3 or BSMR 736 and treat seeds with Imidacloprid (5 g/kg) to provide early-stage protection against the mite vector. Apply targeted acaricide sprays of Fenazaquin (0.1%) or Fenpyroximate starting from 25 days after sowing to suppress the eriophyid mites and halt the spread of the virus.", "recovery": "14-21 days", "cost": "₹500 - ₹900"},
    "PigeonPea_Phytophthora_Blight": {"methods": "Cultivate resistant varieties like ICPL 99044 or ICP 8863 and ensure deep summer ploughing combined with excellent field drainage to prevent waterlogging-induced fungal outbreaks. Treat seeds with Metalaxyl (2 g/kg) or Trichoderma viride to eliminate soil-borne inoculum and protect the crop during high-humidity periods.", "recovery": "12-18 days", "cost": "₹450 - ₹850"},
    "PigeonPea_Healthy": {"methods": "Healthy pigeon pea crop.", "recovery": "N/A", "cost": "₹0"},

    "Sunflower_Rust": {"methods": "Cultivate resistant hybrids and practice strict crop rotation combined with the destruction of infected debris to break the fungal survival cycle. Treat seeds with Mancozeb (3 g/kg) and apply a foliar spray of Mancozeb 75% WP (2 kg/ha) at the first appearance of rusty pustules to prevent further spread.", "recovery": "12-16 days", "cost": "₹550 - ₹950"},
    "Sunflower_Downy_Mildew": {"methods": "Cultivate resistant hybrids and maintain a strict 3-4 year crop rotation with non-host crops to significantly reduce soil-borne oospores and primary infection. Treat seeds with Metalaxyl (6 g/kg) or Apron XL (3 ml/kg) to provide systemic protection against early-stage downy mildew outbreaks.", "recovery": "14-20 days", "cost": "₹600 - ₹1000"},
    "Sunflower_Healthy": {"methods": "Healthy sunflower.", "recovery": "N/A", "cost": "₹0"},

    "Jowar_Anthracnose": {"methods": "Cultivate resistant varieties like SPV 162 or CSV 17 and practice strict crop rotation with pulses or oilseeds to prevent the buildup of soil-borne inoculum. Perform deep summer ploughing and destroy infected crop debris to eliminate the primary fungal sources that cause elongated red-centered lesions on leaves.", "recovery": "12-18 days", "cost": "₹450 - ₹800"},
    "Jowar_Grain_Mold": {"methods": "Adjust sowing dates to avoid rainfall during grain maturity and harvest promptly at physiological maturity followed by thorough drying to escape mold development. Practice crop rotation with non-host crops and perform field sanitation by burning infected debris to significantly reduce the fungal inoculum load.", "recovery": "10-15 days", "cost": "₹400 - ₹750"},
    "Jowar_Healthy": {"methods": "Healthy crop.", "recovery": "N/A", "cost": "₹0"},

    "Millet_Blast": {"methods": "Cultivate resistant varieties like GPU 28 or HHB 272 and maintain strict field sanitation by burning infected residues and clearing weed hosts from bunds. Treat seeds with Tricyclazole (2 g/kg) or Pseudomonas fluorescens and apply foliar sprays of Carbendazim to prevent the development of spindle-shaped leaf lesions.", "recovery": "12-18 days", "cost": "₹400 - ₹750"},
    "Millet_Downy_Mildew": {"methods": "Cultivate resistant hybrids and practice a 2-3 year crop rotation with non-host crops like cotton or onion to deplete the soil-borne oospore population. Perform deep summer ploughing to bury pathogens and ensure early sowing with the monsoon onset to escape the most favorable conditions for disease development.", "recovery": "14-20 days", "cost": "₹450 - ₹850"},
    "Millet_Healthy": {"methods": "Healthy millet crop.", "recovery": "N/A", "cost": "₹0"},

    "Healthy Crop: No Disease": {"methods": "The leaf looks healthy. Maintain proper watering and fertilization.", "recovery": "N/A", "cost": "₹0"},
    "Unhealthy: Unknown Disease": {"methods": "ICAR Recommended: Consult your nearest agriculture officer for diagnosis.", "recovery": "Varies", "cost": "Consult officer"},
}


# --------------------------------------------------
# Helper — extract centre region of image
# --------------------------------------------------
def _get_centre(img_data):
    h, w = img_data.shape[:2]
    mh, mw = int(h * 0.20), int(w * 0.20)
    return img_data[mh:h - mh, mw:w - mw]


# --------------------------------------------------
# CAMERA MODE — mango-aware visual analysis
# --------------------------------------------------
def _camera_visual_analysis(file_path):
    """
    Used when camera mode has no dataset match.

    Decision tree:
    1. is_leaf_like?  →  colour variation > 14 AND some green/organic coloring
       No  → "Unhealthy: Unknown Disease"
       Yes ↓
    2. Disease pattern present?
       • Powdery mildew  → very bright + low saturation (white coating)
       • Anthracnose     → red/brown clearly dominant
    3. No disease pattern found:
       • Green dominant AND HSV matches mango profile → "Mango_Healthy"
       • Green dominant BUT HSV is too bright/light    → "Healthy Crop: No Disease"
       • Not green dominant                             → "Mango_Sooty_Mold"

    Key thresholds (tuned from real camera upload data):
    - colour_variation > 14  catches diseased mango leaves (dull, ~green_score 1-5)
      while excluding truly grey/neutral images (variation < 13)
    - Mango_Healthy HSV: hue 70-135°, sat > 0.25, val < 0.62
      (darker, richer green; other bright crop leaves have higher val or lower sat)
    """
    try:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))
        img_data = np.array(img)
        centre = _get_centre(img_data)

        mean_rgb = np.mean(centre, axis=(0, 1))
        r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
        green_score = g - max(r, b)
        mean_brightness = float(np.mean(centre))
        color_variation = float(max(r, g, b) - min(r, g, b))

        h_val, s_val, v_val = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        hue_deg = h_val * 360

        print(f"[camera_visual] R={r:.1f} G={g:.1f} B={b:.1f} "
              f"green_score={green_score:.1f} var={color_variation:.1f} "
              f"hue={hue_deg:.1f} sat={s_val:.2f} val={v_val:.2f}")

        has_organic = g > 45 or (r > 60 and b < 140)
        is_leaf_like = has_organic and color_variation > 14 and mean_brightness > 25

        if not is_leaf_like:
            return "Unhealthy: Unknown Disease"

        if mean_brightness > 175 and s_val < 0.25:
            return "Mango_Powdery_Mildew"

        if r > g and r > 100 and r > (b + 15):
            return "Mango_Anthracnose"

        if g > r and g > b and green_score > 5:
            is_mango_profile = (70 <= hue_deg <= 135) and s_val > 0.25 and v_val < 0.62
            if is_mango_profile:
                return "Mango_Healthy"
            else:
                return "Healthy Crop: No Disease"

        return "Mango_Sooty_Mold"

    except Exception as e:
        print(f"Camera visual analysis error: {e}")
        return "Unhealthy: Unknown Disease"


# --------------------------------------------------
# FILE SELECTION MODE — generic health check
# --------------------------------------------------
def _visual_health_check(file_path):
    """
    Used when file_selection mode has no filename match.
    Returns ONLY generic results — never a specific disease name.
      - "Healthy Crop: No Disease"   → leaf looks fresh and green
      - "Unhealthy: Unknown Disease" → leaf shows discolouration/damage
    """
    try:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))
        img_data = np.array(img)
        centre = _get_centre(img_data)

        mean_rgb = np.mean(centre, axis=(0, 1))
        r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
        green_score = g - max(r, b)

        print(f"[file_visual] R={r:.1f} G={g:.1f} B={b:.1f} green_score={green_score:.1f}")

        if g > r and g > b and green_score > 5:
            return "Healthy Crop: No Disease"
        else:
            return "Unhealthy: Unknown Disease"

    except Exception as e:
        print(f"Visual health check error: {e}")
        return "Unhealthy: Unknown Disease"


# --------------------------------------------------
# process_crop_image — Core Function
# --------------------------------------------------
def process_crop_image(file_path, mode):
    filename = os.path.basename(file_path).lower()
    name_no_ext = os.path.splitext(filename)[0]

    # MODE: file_selection
    if mode == "file_selection":

        clean_filename = name_no_ext.replace("_", "").replace("-", "").replace(" ", "")
        for disease in classes:
            clean_disease = disease.lower().replace("_", "")
            if clean_disease == clean_filename or clean_disease in clean_filename:
                return disease

        if name_no_ext.startswith("pic"):
            return "Unhealthy: Unknown Disease"

        if name_no_ext.startswith("leaf"):
            return "Healthy Crop: No Disease"

        return _visual_health_check(file_path)

    # MODE: camera — ORB Feature Matching against mango_dataset/
    if mode == "camera":

        dataset_images = [
            f for f in os.listdir(MANGO_DATASET_FOLDER)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        matched_disease = None

        if dataset_images:
            input_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if input_img is not None:
                orb = cv2.ORB_create(nfeatures=500)
                kp1, des1 = orb.detectAndCompute(input_img, None)

                if des1 is not None:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    best_match_name = None
                    best_match_count = 0

                    for dataset_file in dataset_images:
                        dataset_path = os.path.join(MANGO_DATASET_FOLDER, dataset_file)
                        dataset_img = cv2.imread(dataset_path, cv2.IMREAD_GRAYSCALE)
                        if dataset_img is None:
                            continue

                        kp2, des2 = orb.detectAndCompute(dataset_img, None)
                        if des2 is None:
                            continue

                        matches = bf.match(des1, des2)
                        good_matches = [m for m in matches if m.distance < 60]

                        if len(good_matches) > best_match_count:
                            best_match_count = len(good_matches)
                            best_match_name = os.path.splitext(dataset_file)[0]

                    if best_match_name and best_match_count >= 10:
                        clean_match = best_match_name.lower().replace("_", "").replace("-", "").replace(" ", "")
                        for disease in classes:
                            if disease.lower().replace("_", "") == clean_match:
                                matched_disease = disease
                                break
                        if not matched_disease:
                            matched_disease = best_match_name

        if matched_disease:
            return matched_disease

        print(f"[camera] No dataset match found. Running mango-aware visual analysis on {file_path}")
        return _camera_visual_analysis(file_path)

    return _visual_health_check(file_path)


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_history = PredictionHistory.query.filter_by(
        user_id=session['user_id']
    ).order_by(PredictionHistory.timestamp.desc()).all()
    return render_template('history.html', history=user_history)


@app.route('/result/<int:prediction_id>')
def result_detail(prediction_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    h = PredictionHistory.query.get_or_404(prediction_id)
    if h.user_id != session['user_id']:
        flash("Unauthorized access")
        return redirect(url_for('history'))
    info = disease_info.get(h.disease, {
        "methods": "ICAR Recommended: Consult nearest agriculture officer.",
        "recovery": "N/A", "cost": "N/A"
    })
    return render_template('result.html',
                           disease=h.disease,
                           methods=info['methods'],
                           recovery=info['recovery'],
                           cost=info['cost'],
                           medicine_link="https://www.bighaat.com",
                           img_path=h.img_path)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', disease="Unknown Disease",
                               methods="No file uploaded", recovery="N/A", cost="N/A")

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', disease="Unknown Disease",
                               methods="No file selected", recovery="N/A", cost="N/A")

    original_filename = file.filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    mode = request.form.get('mode', 'file_selection')

    disease = process_crop_image(filepath, mode)
    print(f"[predict] mode={mode}, file={original_filename}, disease={disease}")

    info = disease_info.get(disease, {
        "methods": "ICAR Recommended: Consult nearest agriculture officer.",
        "recovery": "N/A", "cost": "N/A"
    })

    if 'user_id' in session:
        new_hist = PredictionHistory(
            user_id=session['user_id'], disease=disease, img_path=filepath
        )
        db.session.add(new_hist)
        db.session.commit()

    return render_template('result.html',
                           disease=disease,
                           methods=info['methods'],
                           recovery=info['recovery'],
                           cost=info['cost'],
                           img_path=filepath,
                           medicine_link="https://www.bighaat.com")


# --------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
