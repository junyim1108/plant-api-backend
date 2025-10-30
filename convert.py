# convert_models_to_tflite.py

import tensorflow as tf
import os
from pathlib import Path


def convert_keras_to_tflite(model_path, output_path, model_name, optimize=True):
    """
    Convert Keras .h5 model to TensorFlow Lite

    Args:
        model_path: Path to .h5 model file
        output_path: Directory to save .tflite file
        model_name: Name for the output file
        optimize: Whether to optimize the model (reduce size)
    """
    print(f"\n{'=' * 60}")
    print(f"Converting {model_name}...")
    print(f"{'=' * 60}")

    try:
        # Load the Keras model
        print(f"📂 Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully!")

        # Print model summary
        print("\n📊 Model Summary:")
        model.summary()

        # Get input shape
        input_shape = model.input_shape
        print(f"\n📐 Input shape: {input_shape}")

        # Convert to TensorFlow Lite
        print(f"\n🔄 Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if optimize:
            print("⚙️ Optimization enabled (smaller size, slightly lower accuracy)")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        else:
            print("⚙️ No optimization (larger size, full accuracy)")

        # Convert
        tflite_model = converter.convert()

        # Save the model
        os.makedirs(output_path, exist_ok=True)
        tflite_filename = f"{model_name}.tflite"
        tflite_path = os.path.join(output_path, tflite_filename)

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        # Get file sizes
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
        reduction = ((original_size - tflite_size) / original_size) * 100

        print(f"\n✅ Conversion successful!")
        print(f"📦 Original model size: {original_size:.2f} MB")
        print(f"📦 TFLite model size: {tflite_size:.2f} MB")
        print(f"📉 Size reduction: {reduction:.1f}%")
        print(f"💾 Saved to: {tflite_path}")

        return tflite_path

    except Exception as e:
        print(f"\n❌ Error converting {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def convert_yolo_to_tflite(yolo_model_path, output_path):
    """
    Convert YOLO11 model to TensorFlow Lite using Ultralytics

    Args:
        yolo_model_path: Path to YOLO .pt model file
        output_path: Directory to save converted model
    """
    print(f"\n{'=' * 60}")
    print(f"Converting YOLO11 to TFLite...")
    print(f"{'=' * 60}")

    try:
        from ultralytics import YOLO

        # Load YOLO model
        print(f"📂 Loading YOLO model from: {yolo_model_path}")
        model = YOLO(yolo_model_path)
        print(f"✅ YOLO model loaded successfully!")

        # Export to TFLite
        print(f"\n🔄 Exporting to TensorFlow Lite...")
        print("⚠️ This may take a few minutes...")

        # Export (Ultralytics handles the conversion)
        export_path = model.export(format='tflite', imgsz=640)

        print(f"\n✅ YOLO conversion successful!")
        print(f"💾 Saved to: {export_path}")

        # Move to output directory if needed
        if output_path and os.path.dirname(export_path) != output_path:
            import shutil
            os.makedirs(output_path, exist_ok=True)
            final_path = os.path.join(output_path, 'yolo11_leaf_detector.tflite')
            shutil.move(export_path, final_path)
            print(f"📦 Moved to: {final_path}")
            return final_path

        return export_path

    except ImportError:
        print("\n❌ Ultralytics not installed!")
        print("Install with: pip install ultralytics")
        return None
    except Exception as e:
        print(f"\n❌ Error converting YOLO: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_labels_file(class_names, output_path, filename='labels.txt'):
    """
    Create labels.txt file for TFLite model

    Args:
        class_names: List of class names
        output_path: Directory to save labels.txt
        filename: Name of the labels file
    """
    labels_path = os.path.join(output_path, filename)

    with open(labels_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    print(f"\n✅ Labels file created: {labels_path}")
    return labels_path


def main():
    """
    Main conversion script
    """
    print("\n" + "=" * 60)
    print("🌿 PlantGuard Model Conversion Script")
    print("=" * 60)

    # UPDATED PATHS - Based on your actual directory structure
    base_path = r"C:\Users\junyi\PycharmProjects\leaf"
    output_path = r"C:\Users\junyi\StudioProjects\plant\assets\ml_models"  # Flutter project

    # Model paths
    efficientnet_path = os.path.join(base_path, "efficientnetv2_complete", "model.h5")
    resnet_path = os.path.join(base_path, "resnet50_complete", "model.h5")
    yolo_path = os.path.join(base_path, "runs", "detect", "yolo11s_leaf_detector", "weights", "best.pt")

    print(f"\n📂 Base path: {base_path}")
    print(f"📂 Output path: {output_path}")

    # Check if models exist
    print("\n📋 Checking model files...")

    models_to_check = {
        "EfficientNetV2": efficientnet_path,
        "ResNet50": resnet_path,
        "YOLO11": yolo_path,
    }

    models_exist = {}
    for model_name, path in models_to_check.items():
        exists = os.path.exists(path)
        models_exist[model_name] = exists
        status = "✅ Found" if exists else "❌ Not found"
        print(f"{status}: {model_name}")
        if not exists:
            print(f"   Expected at: {path}")

    if not any(models_exist.values()):
        print("\n❌ No models found! Please check your paths.")
        print("\n💡 Current paths being checked:")
        for name, path in models_to_check.items():
            print(f"  {name}: {path}")
        return

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    print(f"\n✅ Output directory created: {output_path}")

    converted_models = {}

    # Convert EfficientNetV2
    if models_exist["EfficientNetV2"]:
        result = convert_keras_to_tflite(
            model_path=efficientnet_path,
            output_path=output_path,
            model_name="plant_disease_efficientnet",
            optimize=True
        )
        if result:
            converted_models["EfficientNetV2"] = result

    # Convert ResNet50
    if models_exist["ResNet50"]:
        result = convert_keras_to_tflite(
            model_path=resnet_path,
            output_path=output_path,
            model_name="plant_disease_resnet50",
            optimize=True
        )
        if result:
            converted_models["ResNet50"] = result

    # Convert YOLO11
    if models_exist["YOLO11"]:
        result = convert_yolo_to_tflite(
            yolo_model_path=yolo_path,
            output_path=output_path
        )
        if result:
            converted_models["YOLO11"] = result

    # Create labels file
    print("\n" + "=" * 60)
    print("Creating labels file...")
    print("=" * 60)

    # TODO: Replace with your actual disease class names
    disease_classes = [
        "Healthy",
        "Disease_Class_1",
        "Disease_Class_2",
        "Disease_Class_3",
        # Add all your disease classes here
    ]

    create_labels_file(disease_classes, output_path, 'labels.txt')

    # Summary
    print("\n" + "=" * 60)
    print("🎉 CONVERSION COMPLETE!")
    print("=" * 60)

    if converted_models:
        print("\n✅ Converted Models:")
        for model_name, path in converted_models.items():
            print(f"  • {model_name}")
            print(f"    {path}")
    else:
        print("\n❌ No models were converted!")

    print(f"\n📁 All models saved to: {output_path}")
    print("\n📋 Next steps:")
    print("  1. Update labels.txt with your actual disease class names")
    print("  2. The models are already in your Flutter assets folder")
    print("  3. Make sure pubspec.yaml includes:")
    print("     assets:")
    print("       - assets/ml_models/")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
