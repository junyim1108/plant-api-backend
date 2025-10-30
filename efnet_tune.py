# best_of_both_efficientnetv2.py
import tensorflow as tf
import os
import json
from datetime import datetime

print("🌱 EfficientNetV2 Plant Classifier - Complete & Clean")
print("=" * 55)

DATASET_PATH = r"C:\Users\junyi\PycharmProjects\leaf\data"


def analyze_dataset():
    """Complete dataset analysis"""
    all_classes = [d for d in os.listdir(DATASET_PATH)
                   if os.path.isdir(os.path.join(DATASET_PATH, d))]

    class_stats = {}
    total_images = 0

    print(f"📊 DATASET ANALYSIS ({len(all_classes)} classes):")

    for i, class_name in enumerate(sorted(all_classes), 1):
        class_path = os.path.join(DATASET_PATH, class_name)
        image_count = len([f for f in os.listdir(class_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        class_stats[class_name] = image_count
        total_images += image_count
        clean_name = class_name.replace('___', ' - ').replace('_', ' ')
        print(f"{i:2d}. {clean_name:<45} {image_count:>6} images")

    print(f"\nTotal: {len(all_classes)} classes, {total_images:,} images")
    return all_classes, class_stats


# Analyze dataset
all_classes, stats = analyze_dataset()

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("✅ GPU configured")

# EfficientNetV2 with MINIMAL augmentation
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

print("\n🔧 Setting up MINIMAL augmentation...")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# Load data
train_gen = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='training', shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='validation'
)

num_classes = len(train_gen.class_indices)
print(f"✅ Data loaded: {num_classes} classes")

# Class weights (if imbalanced)
class_weights = None
if stats:
    max_img = max(stats.values())
    min_img = min(stats.values())
    if max_img / min_img > 2.5:
        print("⚖️ Computing class weights for imbalance...")
        class_weights = {}
        total = sum(stats.values())
        for name in train_gen.class_indices:
            if name in stats:
                idx = train_gen.class_indices[name]
                class_weights[idx] = total / (num_classes * stats[name])

# EfficientNetV2 model
print(f"\n🏗️ Building EfficientNetV2 ({num_classes} classes)...")

base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
]

print("\n🚀 Training Phase 1...")
start_time = datetime.now()

history1 = model.fit(train_gen, validation_data=val_gen, epochs=20,
                     callbacks=callbacks, class_weight=class_weights)

phase1_acc = max(history1.history['val_accuracy'])

# Fine-tuning
if phase1_acc > 0.80:
    print("\n🎯 Training Phase 2 (Fine-tuning)...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history2 = model.fit(train_gen, validation_data=val_gen, epochs=10,
                         callbacks=callbacks, class_weight=class_weights)
    final_acc = max(history2.history['val_accuracy'])
else:
    final_acc = phase1_acc

training_time = datetime.now() - start_time

# Save everything
save_dir = "efficientnetv2_complete"
os.makedirs(save_dir, exist_ok=True)

model.save(f'{save_dir}/model.h5')
with open(f'{save_dir}/classes.json', 'w') as f:
    json.dump(list(train_gen.class_indices.keys()), f)

# Save training info
info = {
    'architecture': 'EfficientNetV2B0',
    'augmentation': 'Minimal',
    'classes': list(train_gen.class_indices.keys()),
    'num_classes': num_classes,
    'final_accuracy': float(final_acc),
    'training_time': str(training_time),
    'class_stats': stats
}

with open(f'{save_dir}/training_info.json', 'w') as f:
    json.dump(info, f, indent=2)

print(f"\n🎉 TRAINING COMPLETE!")
print(f"🏆 Final Accuracy: {final_acc * 100:.1f}%")
print(f"⏱️ Training Time: {training_time}")
print(f"💾 Saved to: {save_dir}/")
print("🌱 Ready for real-world testing!")
