# app.py - Plant Disease Detection API v3.0 - ResNet50 Only - PRODUCTION READY
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import cv2
import numpy as np
from PIL import Image
import io
import os
import json

app = Flask(__name__)
CORS(app)


# ========================================
# ✅ NUMPY TYPE CONVERTER (FIX FOR JSON SERIALIZATION)
# ========================================
def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# ========================================
# ✅ LOAD TREATMENT DATABASE
# ========================================
TREATMENT_DB = {}
try:
    treatment_db_path = 'treatment_database.json'
    if os.path.exists(treatment_db_path):
        with open(treatment_db_path, 'r', encoding='utf-8') as f:
            TREATMENT_DB = json.load(f)
        print(f"✅ Treatment database loaded: {len(TREATMENT_DB)} diseases with detailed info")
    else:
        print("⚠️ treatment_database.json not found - using fallback disease info")
except Exception as e:
    print(f"❌ Error loading treatment database: {e}")

# ========================================
# ✅ LOAD MODELS (YOLO11 + ResNet50 ONLY)
# ========================================
print("🔄 Loading AI models...")

# Load YOLO11 for leaf detection
try:
    yolo_model = YOLO('runs/detect/yolo11s_leaf_detector/weights/best.pt')
    print("✅ YOLO11 Leaf Detector loaded!")
except Exception as e:
    print(f"⚠️ YOLO11 load failed: {e}")
    yolo_model = None

# Load ResNet50 for disease classification
try:
    resnet_model = load_model('resnet50_complete/model.h5')
    print("✅ ResNet50 Disease Classifier loaded!")
except Exception as e:
    print(f"❌ ResNet50 load failed: {e}")
    resnet_model = None
    raise Exception("❌ CRITICAL: ResNet50 model required but failed to load!")

# ========================================
# ✅ DISEASE CLASSES (66 total)
# ========================================
DISEASE_CLASSES = [
    "Apple___alternaria_leaf_spot", "Apple___black_rot", "Apple___brown_spot",
    "Apple___gray_spot", "Apple___healthy", "Apple___rust", "Apple___scab",
    "Bell_pepper___bacterial_spot", "Bell_pepper___healthy", "Blueberry___healthy",
    "Cassava___bacterial_blight", "Cassava___brown_streak_disease", "Cassava___green_mottle",
    "Cassava___healthy", "Cassava___mosaic_disease", "Cherry___healthy",
    "Cherry___powdery_mildew", "Coffee___healthy", "Coffee___red_spider_mite",
    "Coffee___rust", "Grape___Leaf_blight", "Grape___black_measles", "Grape___black_rot",
    "Grape___healthy", "Orange___citrus_greening", "Peach___bacterial_spot",
    "Peach___healthy", "Potato___bacterial_wilt", "Potato___early_blight",
    "Potato___healthy", "Potato___late_blight", "Potato___leafroll_virus",
    "Potato___mosaic_virus", "Potato___nematode", "Potato___pests",
    "Potato___phytophthora", "Raspberry___healthy", "Rice___bacterial_blight",
    "Rice___blast", "Rice___brown_spot", "Rice___tungro", "Rose___healthy",
    "Rose___rust", "Rose___slug_sawfly", "Soybean___healthy",
    "Squash___powdery_mildew", "Strawberry___healthy", "Strawberry___leaf_scorch",
    "Sugercane___healthy", "Sugercane___mosaic", "Sugercane___rust",
    "Sugercane___yellow_leaf", "Tomato___bacterial_spot", "Tomato___early_blight",
    "Tomato___healthy", "Tomato___late_blight", "Tomato___leaf_curl",
    "Tomato___leaf_mold", "Tomato___mosaic_virus", "Tomato___septoria_leaf_spot",
    "Tomato___spider_mites", "Tomato___target_spot", "Watermelon___anthracnose",
    "Watermelon___downy_mildew", "Watermelon___healthy", "Watermelon___mosaic_virus"
]


# ========================================
# ✅ HELPER FUNCTIONS
# ========================================
def preprocess_for_resnet(img):
    """ResNet50 preprocessing"""
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = resnet_preprocess(img)
    return np.expand_dims(img, axis=0)


def get_severity_level(confidence):
    """Determine disease severity based on confidence"""
    if confidence >= 0.85:
        return 'severe'
    elif confidence >= 0.65:
        return 'moderate'
    else:
        return 'mild'


def get_treatment_info(disease_class, confidence):
    """Get comprehensive treatment information"""
    severity = get_severity_level(confidence)

    # Try to get from treatment database
    if disease_class in TREATMENT_DB:
        disease_info = TREATMENT_DB[disease_class]
        severity_data = disease_info.get('severity_levels', {}).get(severity, {})

        # Fallback to any available severity if exact not found
        if not severity_data and disease_info.get('severity_levels'):
            severity_data = list(disease_info['severity_levels'].values())[0]

        return {
            'available': True,
            'disease_name': disease_info.get('disease_name', ''),
            'scientific_name': disease_info.get('scientific_name', ''),
            'description': disease_info.get('description', ''),
            'severity': severity,
            'symptoms': severity_data.get('symptoms', ''),
            'treatment_steps': severity_data.get('treatment_steps', []),
            'prevention': severity_data.get('prevention', []),
            'recovery_time': severity_data.get('recovery_time', 'Varies'),
            'success_rate': severity_data.get('success_rate', 'N/A'),
            'cost': severity_data.get('cost', 'Varies'),
            'sources': severity_data.get('sources', []),
            'requires_professional': severity_data.get('professional_consultation', False),
            'organic_alternatives': disease_info.get('organic_alternatives', {}),
            'environmental_conditions': disease_info.get('environmental_conditions', {})
        }

    # Fallback for diseases not in database yet
    return {
        'available': False,
        'message': 'Detailed treatment information being researched. Please consult agricultural extension service.',
        'severity': severity
    }


def assess_image_quality(img_array):
    """Assess input image quality"""
    # Convert to grayscale for analysis
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array

    # Calculate sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Calculate brightness
    brightness = np.mean(gray)

    quality = {
        'sharpness': float(laplacian_var),
        'brightness': float(brightness),
        'is_blurry': bool(laplacian_var < 100),
        'is_too_dark': bool(brightness < 50),
        'is_too_bright': bool(brightness > 200),
        'quality_score': 'good'
    }

    if quality['is_blurry']:
        quality['quality_score'] = 'blurry'
    elif quality['is_too_dark']:
        quality['quality_score'] = 'too_dark'
    elif quality['is_too_bright']:
        quality['quality_score'] = 'too_bright'

    return quality


# ========================================
# ✅ API ENDPOINTS
# ========================================
@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Plant Disease Detection API v3.0 🌿',
        'version': '3.0 - ResNet50 Only',
        'total_classes': len(DISEASE_CLASSES),
        'treatment_database_size': len(TREATMENT_DB),
        'pipeline': 'YOLO11 Leaf Detection → ResNet50 Classification',
        'models_loaded': {
            'yolo11': yolo_model is not None,
            'resnet50': resnet_model is not None,
        },
        'features': [
            'STRICT leaf detection with YOLO11 (rejects non-leaf images)',
            'ResNet50 disease classification',
            'Treatment database with scientific citations',
            'Top-3 predictions',
            'Image quality assessment',
            'Confidence-based severity scoring'
        ]
    })


@app.route('/detect', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict_disease():
    """Main prediction endpoint - ResNet50 Only with Strict YOLO"""
    try:
        # ✅ STEP 1: Validate request
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400

        # ✅ STEP 2: Read and convert image
        file = request.files['image']
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)

        # Convert to BGR
        if len(img_array.shape) == 2:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[-1] == 4:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # ✅ STEP 3: Assess image quality
        quality = assess_image_quality(img_bgr)
        if quality['quality_score'] != 'good':
            print(f"⚠️ Image quality warning: {quality['quality_score']}")

        # ✅ STEP 4: STRICT YOLO Leaf Detection (RAISES THRESHOLD TO 50%)
        leaf_confidence = 0.0
        YOLO_CONFIDENCE_THRESHOLD = 0.60  # ✅ STRICT: Raised from 0.25 to 0.50

        if yolo_model is not None:
            try:
                results = yolo_model(img_bgr, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)

                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    best_idx = np.argmax(confidences)

                    x1, y1, x2, y2 = map(int, boxes[best_idx])
                    leaf_confidence = float(confidences[best_idx])
                    leaf_img = img_bgr[y1:y2, x1:x2]

                    print(f"🍃 Leaf detected with confidence: {leaf_confidence:.2%}")

                    # ✅ DOUBLE-CHECK: Even if detected, reject if too low
                    if leaf_confidence < YOLO_CONFIDENCE_THRESHOLD:
                        print(f"❌ Leaf confidence too low: {leaf_confidence:.2%}")
                        return jsonify({
                            'success': False,
                            'error': 'Leaf detection confidence too low',
                            'message': f'Please take a clearer photo of a real plant leaf. '
                                       f'Detected confidence: {leaf_confidence:.1%}',
                            'leaf_detected': False,
                            'leaf_confidence': round(leaf_confidence * 100, 2),
                            'image_quality': quality
                        }), 400
                else:
                    print("❌ No leaf detected in image")
                    return jsonify({
                        'success': False,
                        'error': 'No leaf detected',
                        'message': 'No plant leaf found in the image. Please ensure:\n'
                                   '• The image contains a REAL plant leaf (not a screen/photo)\n'
                                   '• The leaf is clearly visible and in focus\n'
                                   '• The image has good lighting\n'
                                   '• Avoid reflections or blurry images',
                        'leaf_detected': False,
                        'image_quality': quality
                    }), 400
            except Exception as e:
                print(f"⚠️ YOLO error: {e}")
                # If YOLO fails, reject the image (don't proceed)
                return jsonify({
                    'success': False,
                    'error': 'Leaf detection failed',
                    'message': 'Could not process image for leaf detection. Please try again with a clear leaf photo.',
                    'leaf_detected': False,
                }), 400
        else:
            # ✅ If YOLO not available, use full image but warn
            print("⚠️ YOLO model not available - using full image")
            leaf_img = img_bgr

        # ✅ STEP 5: ResNet50 Disease Classification
        print("🎯 Running ResNet50 classification...")
        processed_img = preprocess_for_resnet(leaf_img)
        predictions = resnet_model.predict(processed_img, verbose=0)[0]
        model_used = "ResNet50"

        # ✅ STEP 6: Get top-3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = []

        for idx in top_3_indices:
            disease_class = DISEASE_CLASSES[idx]
            confidence = float(predictions[idx])

            # Parse disease name
            parts = disease_class.split("___")
            plant_type = parts[0].replace("_", " ").title()
            disease_name = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"

            top_3_predictions.append({
                'rank': len(top_3_predictions) + 1,
                'plant_type': plant_type,
                'disease': disease_name,
                'disease_class': disease_class,
                'confidence': round(confidence * 100, 2),
                'is_healthy': bool('healthy' in disease_class.lower())
            })

        # Primary prediction
        primary = top_3_predictions[0]
        print(f"✅ Primary: {primary['plant_type']} - {primary['disease']} ({primary['confidence']}%)")

        # ✅ STEP 7: Get treatment information
        treatment_info = get_treatment_info(primary['disease_class'], primary['confidence'] / 100)

        # ✅ STEP 8: Calculate severity
        if primary['is_healthy']:
            severity = "None"
        else:
            severity = get_severity_level(primary['confidence'] / 100).title()

        # ✅ STEP 9: Build response
        response_data = {
            'success': True,
            'model_used': model_used,
            'plant_type': primary['plant_type'],
            'disease': primary['disease'],
            'confidence': float(primary['confidence']),
            'severity': severity,
            'is_healthy': bool(primary['is_healthy']),

            # Leaf detection info
            'leaf_detected': bool(leaf_confidence > 0.0),
            'leaf_confidence': round(leaf_confidence * 100, 2) if leaf_confidence > 0.0 else None,

            # Image quality
            'image_quality': quality,

            # Treatment information
            'treatment': treatment_info,

            # Top 3 predictions
            'top_predictions': top_3_predictions,

            # Legacy fields for backward compatibility
            'symptoms': treatment_info.get('symptoms', 'Consult specialist for diagnosis'),
            'prevention': treatment_info.get('prevention', [])
        }

        # ✅ STEP 10: Convert all NumPy types to native Python types
        response_data = convert_numpy_types(response_data)

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ========================================
# ✅ RUN SERVER
# ========================================
if __name__ == '__main__':
    import os

    os.makedirs('uploads', exist_ok=True)

    print("\n" + "=" * 70)
    print("🌿 Plant Disease Detection API v3.0")
    print("=" * 70)
    print(f"📊 Disease classes: {len(DISEASE_CLASSES)}")
    print(f"📚 Treatment database: {len(TREATMENT_DB)} diseases")
    print(f"🤖 YOLO11 (Strict 60% threshold): {'✅' if yolo_model else '❌'}")
    print(f"🧠 ResNet50: {'✅' if resnet_model else '❌'}")
    print("=" * 70)
    print("\n🚀 Starting Flask server...")
    print("📡 Server running at: http://localhost:5000")
    print("⚠️  Keep this terminal open while using the app!\n")

    # Start Flask
    app.run(host='0.0.0.0', port=5000, debug=False)


