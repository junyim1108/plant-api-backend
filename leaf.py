# leaf_disease_detector_with_gui.py
from ultralytics import YOLO
import tensorflow as tf
import cv2
import numpy as np
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import threading


class LeafDiseaseDetectorGUI:
    def __init__(self):
        print("🔄 Loading models...")

        # Load YOLO for leaf detection
        try:
            self.yolo = YOLO("runs/detect/yolo11s_leaf_detector/weights/best.pt")
            print("✅ YOLO leaf detector loaded")
        except Exception as e:
            print(f"❌ YOLO loading failed: {e}")
            exit(1)

        # Load disease classification models
        self.efficientnet_model = None
        self.resnet_model = None
        self.efficientnet_classes = []
        self.resnet_classes = []

        self.load_disease_models()

        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("🍃 Leaf Disease Detection System")
        self.root.geometry("600x400")
        self.root.configure(bg='#f0f0f0')

        self.setup_gui()

    def load_disease_models(self):
        """Load disease classification models"""
        # Load EfficientNetV2
        try:
            self.efficientnet_model = tf.keras.models.load_model('efficientnetv2_complete/model.h5')
            with open('efficientnetv2_complete/classes.json', 'r') as f:
                self.efficientnet_classes = json.load(f)
            print(f"✅ EfficientNetV2 loaded ({len(self.efficientnet_classes)} classes)")
        except Exception as e:
            print(f"⚠️ EfficientNetV2 not available: {e}")

        # Load ResNet50
        try:
            self.resnet_model = tf.keras.models.load_model('resnet50_complete/model.h5')
            with open('resnet50_complete/classes.json', 'r') as f:
                self.resnet_classes = json.load(f)
            print(f"✅ ResNet50 loaded ({len(self.resnet_classes)} classes)")
        except Exception as e:
            print(f"⚠️ ResNet50 not available: {e}")

        if not self.efficientnet_model and not self.resnet_model:
            print("❌ No disease classification models available!")
            exit(1)

    def setup_gui(self):
        """Setup the GUI interface"""
        # Title
        title_label = tk.Label(
            self.root,
            text="🍃 Leaf Disease Detection System",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#2d5a27'
        )
        title_label.pack(pady=20)

        # Instructions
        instruction_label = tk.Label(
            self.root,
            text="Click the button below to select an image file",
            font=("Arial", 12),
            bg='#f0f0f0'
        )
        instruction_label.pack(pady=10)

        # Select Image Button
        self.select_button = tk.Button(
            self.root,
            text="📁 Select Image File",
            font=("Arial", 14, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=30,
            pady=15,
            command=self.select_and_analyze_image
        )
        self.select_button.pack(pady=20)

        # Status Label
        self.status_label = tk.Label(
            self.root,
            text="Ready to analyze leaf images",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#666'
        )
        self.status_label.pack(pady=10)

        # Results Text Area
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(pady=20, padx=20, fill='both', expand=True)

        self.results_text = tk.Text(
            self.results_frame,
            height=15,
            width=70,
            font=("Consolas", 9),
            wrap=tk.WORD
        )

        scrollbar = tk.Scrollbar(self.results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_text.pack(side=tk.LEFT, fill='both', expand=True)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)

    def select_and_analyze_image(self):
        """Select image file and analyze it"""
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select a leaf image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.status_label.config(text=f"Processing: {os.path.basename(file_path)}")
        self.select_button.config(state='disabled')

        # Run analysis in separate thread to prevent GUI freezing
        threading.Thread(target=self.analyze_image, args=(file_path,), daemon=True).start()

    def analyze_image(self, image_path):
        """Analyze the selected image"""
        try:
            # Clear previous results
            self.results_text.delete(1.0, tk.END)

            # Validate image
            if not os.path.exists(image_path):
                self.display_result("❌ Image file not found!")
                return

            image = cv2.imread(image_path)
            if image is None:
                self.display_result("❌ Cannot read image file!")
                return

            self.display_result(f"✅ Analyzing: {os.path.basename(image_path)}\n")

            # Detect leaves
            self.display_result("🔍 Detecting leaves...\n")
            leaves = self.detect_leaves(image_path)

            if not leaves:
                self.display_result("❌ No valid leaves detected!\n")
                self.display_result("💡 Tips:\n")
                self.display_result("   - Ensure leaves are clearly visible\n")
                self.display_result("   - Use good lighting\n")
                self.display_result("   - Move closer to get larger leaf regions\n")
                return

            self.display_result(f"✅ Found {len(leaves)} valid leaf(s)\n\n")

            # Display detected leaves
            self.display_result("📋 Detected Leaves:\n")
            for leaf in leaves:
                self.display_result(f"   Leaf {leaf['id']}: Size {leaf['size']}, Confidence {leaf['confidence']:.3f}\n")

            # Analyze each leaf
            for leaf in leaves:
                self.display_result(f"\n{'=' * 50}\n")
                self.display_result(f"🍃 ANALYZING LEAF {leaf['id']}\n")
                self.display_result(f"{'=' * 50}\n")

                disease_results = self.classify_disease(leaf['crop'])
                self.display_leaf_results(leaf, disease_results)

        except Exception as e:
            self.display_result(f"❌ Error: {str(e)}\n")
        finally:
            # Re-enable button
            self.root.after(0, lambda: self.select_button.config(state='normal'))
            self.root.after(0, lambda: self.status_label.config(text="Analysis complete! Ready for next image."))

    def detect_leaves(self, image_path):
        """Detect leaves in image"""
        image = cv2.imread(image_path)
        results = self.yolo(image_path, conf=0.5, verbose=False)

        leaves = []
        for r in results:
            if r.boxes is not None:
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())

                    width = x2 - x1
                    height = y2 - y1

                    if width >= 100 and height >= 100:
                        leaves.append({
                            'id': len(leaves) + 1,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'size': f"{width}x{height}",
                            'crop': image[y1:y2, x1:x2]
                        })

        return leaves

    def preprocess_for_efficientnet(self, leaf_crop):
        """Preprocess for EfficientNetV2"""
        leaf_resized = cv2.resize(leaf_crop, (224, 224))
        leaf_rgb = cv2.cvtColor(leaf_resized, cv2.COLOR_BGR2RGB)
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        leaf_processed = preprocess_input(leaf_rgb.astype(np.float32))
        return np.expand_dims(leaf_processed, axis=0)

    def preprocess_for_resnet(self, leaf_crop):
        """Preprocess for ResNet50"""
        leaf_resized = cv2.resize(leaf_crop, (224, 224))
        leaf_rgb = cv2.cvtColor(leaf_resized, cv2.COLOR_BGR2RGB)
        from tensorflow.keras.applications.resnet50 import preprocess_input
        leaf_processed = preprocess_input(leaf_rgb.astype(np.float32))
        return np.expand_dims(leaf_processed, axis=0)

    def classify_disease(self, leaf_crop):
        """Classify disease using both models"""
        results = {}

        # EfficientNetV2 prediction
        if self.efficientnet_model is not None:
            try:
                processed = self.preprocess_for_efficientnet(leaf_crop)
                predictions = self.efficientnet_model.predict(processed, verbose=0)

                top_idx = np.argmax(predictions[0])
                top_3_indices = np.argsort(predictions[0])[-3:][::-1]

                results['efficientnet'] = {
                    'top_prediction': {
                        'disease': self.efficientnet_classes[top_idx],
                        'confidence': float(predictions[0][top_idx])
                    },
                    'top_3': [
                        {
                            'disease': self.efficientnet_classes[idx],
                            'confidence': float(predictions[0][idx])
                        }
                        for idx in top_3_indices
                    ]
                }
            except Exception as e:
                results['efficientnet'] = {'error': str(e)}

        # ResNet50 prediction
        if self.resnet_model is not None:
            try:
                processed = self.preprocess_for_resnet(leaf_crop)
                predictions = self.resnet_model.predict(processed, verbose=0)

                top_idx = np.argmax(predictions[0])
                top_3_indices = np.argsort(predictions[0])[-3:][::-1]

                results['resnet'] = {
                    'top_prediction': {
                        'disease': self.resnet_classes[top_idx],
                        'confidence': float(predictions[0][top_idx])
                    },
                    'top_3': [
                        {
                            'disease': self.resnet_classes[idx],
                            'confidence': float(predictions[0][idx])
                        }
                        for idx in top_3_indices
                    ]
                }
            except Exception as e:
                results['resnet'] = {'error': str(e)}

        return results

    def display_leaf_results(self, leaf_info, disease_results):
        """Display leaf analysis results in GUI"""
        self.display_result(f"📍 Location: {leaf_info['bbox']}\n")
        self.display_result(f"📏 Size: {leaf_info['size']}\n")
        self.display_result(f"🎯 YOLO Confidence: {leaf_info['confidence']:.3f}\n\n")

        # EfficientNetV2 Results
        if 'efficientnet' in disease_results:
            self.display_result("🔵 EfficientNetV2 Results:\n")
            eff_result = disease_results['efficientnet']

            if 'error' in eff_result:
                self.display_result(f"   ❌ Error: {eff_result['error']}\n")
            else:
                top_pred = eff_result['top_prediction']
                disease_name = top_pred['disease'].replace('___', ' - ').replace('_', ' ')
                confidence = top_pred['confidence']

                self.display_result(f"   🏆 Top: {disease_name}\n")
                self.display_result(f"   📊 Confidence: {confidence:.3f} ({confidence * 100:.1f}%)\n")

                self.display_result(f"   📋 Top 3:\n")
                for i, pred in enumerate(eff_result['top_3'][:3]):
                    name = pred['disease'].replace('___', ' - ').replace('_', ' ')
                    conf = pred['confidence']
                    self.display_result(f"      {i + 1}. {name} ({conf:.3f})\n")

        # ResNet50 Results
        if 'resnet' in disease_results:
            self.display_result("\n🔴 ResNet50 Results:\n")
            res_result = disease_results['resnet']

            if 'error' in res_result:
                self.display_result(f"   ❌ Error: {res_result['error']}\n")
            else:
                top_pred = res_result['top_prediction']
                disease_name = top_pred['disease'].replace('___', ' - ').replace('_', ' ')
                confidence = top_pred['confidence']

                self.display_result(f"   🏆 Top: {disease_name}\n")
                self.display_result(f"   📊 Confidence: {confidence:.3f} ({confidence * 100:.1f}%)\n")

                self.display_result(f"   📋 Top 3:\n")
                for i, pred in enumerate(res_result['top_3'][:3]):
                    name = pred['disease'].replace('___', ' - ').replace('_', ' ')
                    conf = pred['confidence']
                    self.display_result(f"      {i + 1}. {name} ({conf:.3f})\n")

        # Model Agreement
        if 'efficientnet' in disease_results and 'resnet' in disease_results:
            eff_result = disease_results['efficientnet']
            res_result = disease_results['resnet']

            if 'error' not in eff_result and 'error' not in res_result:
                eff_disease = eff_result['top_prediction']['disease']
                res_disease = res_result['top_prediction']['disease']

                self.display_result("\n🤝 Model Agreement:\n")
                if eff_disease == res_disease:
                    self.display_result(f"   ✅ Both models agree: {eff_disease.replace('_', ' ')}\n")
                else:
                    self.display_result("   ⚠️ Models disagree\n")
                    eff_conf = eff_result['top_prediction']['confidence']
                    res_conf = res_result['top_prediction']['confidence']

                    if eff_conf > res_conf:
                        self.display_result(f"      → EfficientNet more confident ({eff_conf:.3f} vs {res_conf:.3f})\n")
                    else:
                        self.display_result(f"      → ResNet more confident ({res_conf:.3f} vs {eff_conf:.3f})\n")

    def display_result(self, text):
        """Display text in results area"""
        self.root.after(0, lambda: self.results_text.insert(tk.END, text))
        self.root.after(0, lambda: self.results_text.see(tk.END))

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    print("🍃 Starting Leaf Disease Detection GUI...")
    app = LeafDiseaseDetectorGUI()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
