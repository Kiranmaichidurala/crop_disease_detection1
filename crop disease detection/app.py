from flask import Flask, render_template, request
import os
import torch
from PIL import Image
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
DATASET_FOLDER = "mango_dataset"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = SentenceTransformer('clip-ViT-B-32')

dataset_embeddings = []
dataset_filenames = []

# Load mango dataset images
if os.path.exists(DATASET_FOLDER):
    for file in os.listdir(DATASET_FOLDER):
        path = os.path.join(DATASET_FOLDER, file)
        try:
            image = Image.open(path)
            emb = model.encode(image, convert_to_tensor=True)
            dataset_embeddings.append(emb)
            dataset_filenames.append(file)
        except:
            pass


classes = [
"Rice_Bacterial_Blight","Rice_Blast","Rice_Tungro","Rice_Healthy",
"Cotton_Bacterial_Blight","Cotton_Leaf_Curl","Cotton_Wilt","Cotton_Healthy",
"Chili_Leaf_Curl","Chili_Anthracnose","Chili_Bacterial_Spot","Chili_Healthy",
"Maize_Gray_Leaf_Spot","Maize_Leaf_Blight","Maize_Common_Rust","Maize_Healthy",
"Groundnut_Leaf_Spot","Groundnut_Rust","Groundnut_Healthy",
"Turmeric_Leaf_Blight","Turmeric_Leaf_Spot","Turmeric_Healthy",
"Sugarcane_Red_Rot","Sugarcane_Smut","Sugarcane_Healthy",
"Tomato_Late_Blight","Tomato_Early_Blight","Tomato_Leaf_Curl","Tomato_Healthy",
"Papaya_Ring_Spot","Papaya_Mosaic","Papaya_Healthy",
"Mango_Anthracnose","Mango_Powdery_Mildew","Mango_Sooty_Mold","Mango_Healthy",
"Banana_Panama_Disease","Banana_Sigatoka","Banana_Healthy",
"PigeonPea_Sterility_Mosaic","PigeonPea_Phytophthora_Blight","PigeonPea_Healthy",
"Sunflower_Rust","Sunflower_Downy_Mildew","Sunflower_Healthy",
"Jowar_Anthracnose","Jowar_Grain_Mold","Jowar_Healthy",
"Millet_Blast","Millet_Downy_Mildew","Millet_Healthy"
]


preventive_methods = {
"Mango_Anthracnose":"Apply copper-based fungicides and prune affected twigs.",
"Mango_Powdery_Mildew":"Spray sulfur fungicides and reduce humidity.",
"Mango_Sooty_Mold":"Control mealybugs and whiteflies.",
"Mango_Healthy":"Healthy mango tree."

"Rice_Bacterial_Blight": "Use resistant varieties, avoid high nitrogen, keep fields drained.",
    "Rice_Blast": "Use fungicide sprays, maintain field sanitation.",
    "Rice_Tungro": "Control leafhopper insects, remove infected plants.",
    "Rice_Healthy": "The crop is healthy. Maintain good fertilizer and irrigation practices.",

    "Cotton_Bacterial_Blight": "Use resistant seeds, treat seeds before sowing, apply copper sprays.",
    "Cotton_Leaf_Curl": "Use virus-free seeds, control whiteflies, remove infected plants.",
    "Cotton_Wilt": "Improve soil drainage and use resistant cotton varieties.",
    "Cotton_Healthy": "Healthy plant. Continue good soil and water management.",

    "Chili_Leaf_Curl": "Use insect repellents to control whiteflies and aphids.",
    "Chili_Anthracnose": "Apply fungicides and avoid excess humidity.",
    "Chili_Bacterial_Spot": "Use copper sprays and disease-free seeds.",
    "Chili_Healthy": "Plant is healthy. Keep monitoring regularly.",

    "Maize_Gray_Leaf_Spot": "Use resistant hybrids, rotate crops, avoid overhead irrigation.",
    "Maize_Leaf_Blight": "Remove infected leaves and apply recommended fungicides.",
    "Maize_Common_Rust": "Plant rust-resistant varieties and use fungicide if severe.",
    "Maize_Healthy": "Healthy crop. Maintain proper spacing and irrigation.",

    "Groundnut_Leaf_Spot": "Apply protective fungicides and rotate crops.",
    "Groundnut_Rust": "Use resistant groundnut varieties and apply sulfur-based spray.",
    "Groundnut_Healthy": "Healthy plant. Maintain soil fertility.",

    "Turmeric_Leaf_Blight": "Improve air circulation and apply copper fungicide.",
    "Turmeric_Leaf_Spot": "Avoid waterlogging and remove damaged leaves.",
    "Turmeric_Healthy": "Healthy plant. Maintain organic fertilizer supply.",

    "Sugarcane_Red_Rot": "Use resistant sugarcane varieties and treat seed sets.",
    "Sugarcane_Smut": "Remove infected clumps and use disease-free planting material.",
    "Sugarcane_Healthy": "Healthy plant. Maintain good irrigation and fertilizer schedule.",

    "Tomato_Late_Blight": "Destroy infected plants, apply fungicides early.",
    "Tomato_Early_Blight": "Use copper fungicide, remove lower infected leaves.",
    "Tomato_Leaf_Curl": "Control whiteflies and avoid planting near infected crops.",
    "Tomato_Healthy": "Your plant is healthy. Continue proper watering.",

    "Papaya_Ring_Spot": "Control aphids and remove infected papaya plants.",
    "Papaya_Mosaic": "Use virus-free seedlings and control insect vectors.",
    "Papaya_Healthy": "Healthy papaya plant. Maintain nutrient balance.",

    "Mango_Anthracnose": "Apply copper-based fungicides and prune affected twigs.",
    "Mango_Powdery_Mildew": "Spray sulfur fungicides and avoid excess humidity.",
    "Mango_Sooty_Mold": "Control mealybugs and whiteflies that secrete honeydew.",
    "Mango_Healthy": "Healthy mango tree. Continue regular pruning.",

    "Banana_Panama_Disease": "Use resistant varieties, improve soil drainage.",
    "Banana_Sigatoka": "Remove infected leaves and use fungicidal sprays.",
    "Banana_Healthy": "Healthy banana plant. Ensure proper irrigation.",

    "PigeonPea_Sterility_Mosaic": "Control insect vectors and use resistant seeds.",
    "PigeonPea_Phytophthora_Blight": "Improve drainage and avoid waterlogging.",
    "PigeonPea_Healthy": "Healthy pigeon pea crop.",

    "Sunflower_Rust": "Use resistant hybrids and apply fungicides.",
    "Sunflower_Downy_Mildew": "Use treated seeds and rotate crops.",
    "Sunflower_Healthy": "Healthy sunflower.",

    "Jowar_Anthracnose": "Spray fungicides and remove infected residue.",
    "Jowar_Grain_Mold": "Harvest timely and dry grains properly.",
    "Jowar_Healthy": "Healthy crop.",

    "Millet_Blast": "Use resistant varieties and maintain field sanitation.",
    "Millet_Downy_Mildew": "Use disease-free seeds and avoid high moisture.",
    "Millet_Healthy": "Healthy millet crop."
}


def compare_with_dataset(image_path):

    try:
        image = Image.open(image_path)
        test_emb = model.encode(image, convert_to_tensor=True)

        scores = util.cos_sim(test_emb, dataset_embeddings)[0]
        best_score = torch.max(scores)
        best_index = torch.argmax(scores)

        if best_score > 0.75:
            return dataset_filenames[best_index]

    except:
        pass

    return None


def predict_disease_from_filename(filename):

    filename = filename.lower()

    for disease in classes:
        if disease.lower().replace('_','') in filename.replace('_',''):
            return disease

    return "Unknown Disease"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']
    source = request.form.get("source")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)


    # CAMERA → compare with mango dataset
    if source == "camera":

        matched_file = compare_with_dataset(filepath)

        if matched_file:
            disease = os.path.splitext(matched_file)[0]
        else:
            disease = "Unknown Mango Disease"

        methods = preventive_methods.get(
            disease,
            "No preventive information available."
        )


    # FILE UPLOAD → detect by filename
    else:

        disease = predict_disease_from_filename(filename)

        if disease == "Unknown Disease":

            if filename.lower().startswith("leaf"):
                disease = "Healthy Plant"
                methods = "The plant is healthy."

            elif filename.lower().startswith("pic"):
                disease = "Unhealthy Plant - Unknown Disease"
                methods = "Plant appears unhealthy but disease not identified."

            else:
                methods = "No preventive information available."

        else:
            methods = "Preventive information not stored for this crop."


    return render_template(
        'result.html',
        disease=disease,
        methods=methods,
        img_path=filepath
    )


if __name__ == '__main__':
    app.run(debug=True)
