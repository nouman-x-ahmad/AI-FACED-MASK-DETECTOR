#Nouman Ahmad



import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess

# Configuration and parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 3  # with_mask, without_mask, mask_weared_incorrect

# Definin paths to your dataset 
IMAGES_DIR = os.path.join("archive2", "images")
ANNOTATIONS_DIR = os.path.join("archive2", "annotations")

# Map string labels to numeric indices
label_map = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}


def parse_annotation(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find("filename").text
    boxes = []
    labels = []
    for obj in root.iter("object"):
        label = obj.find("name").text
        if label not in label_map:
            continue
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append((xmin, ymin, xmax, ymax))
        labels.append(label_map[label])
    return filename, boxes, labels


def load_dataset():

    images = []
    labels = []
    # Processin each XML file in the annotations directory
    for xml_file in os.listdir(ANNOTATIONS_DIR):
        if not xml_file.endswith(".xml"):
            continue
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
        filename, boxes, lbls = parse_annotation(xml_path)
        img_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(img_path):
            continue
        # Read the image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converting BGR to RGB
        for (xmin, ymin, xmax, ymax), lbl in zip(boxes, lbls):
            # Crop the face region
            face = image[ymin:ymax, xmin:xmax]
            if face.size == 0:
                continue
            # Resize the crop to the desired size
            face = cv2.resize(face, IMAGE_SIZE)
            images.append(face)
            labels.append(lbl)
    images = np.array(images, dtype="float32")
    labels = np.array(labels)
    return images, labels


print("Loading dataset...")
X, y = load_dataset()
print("Total samples:", len(X))

# Normalize images and convert labels to one-hot encoding
X = X / 255.0
y_cat = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

# Split dataset for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Data augmentation generator
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator()  

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# Dictionary mapping model names to their respective pre-trained network and preprocessing
PRETRAINED_MODELS = {
    "MobileNetV2": {
        "base_model": MobileNetV2(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,)),
        "preprocess": mobilenet_preprocess
    },
    "ResNet50": {
        "base_model": ResNet50(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,)),
        "preprocess": resnet_preprocess
    },
    "EfficientNetB0": {
        "base_model": EfficientNetB0(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,)),
        "preprocess": efficient_preprocess
    }
}


def build_model(pretrained, freeze_base=True, fine_tune_at=None):



    # Parameterssssssssssssssss:
    # - pretrained: dictionary with keys "base_model" and "preprocess"
    # - freeze_base: whether to freeze the base model layers for feature extraction.
    # - fine_tune_at: if not None, unfreeze the top layers from this index for fine-tuning.

    base_model = pretrained["base_model"]
    # Freeze base model layers if requested.
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    elif fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        # If not freezing and no fine_tune_at given, all layers are trainable.
        for layer in base_model.layers:
            layer.trainable = True

    # Addingg custom head for our classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# Created dictionaries to store models and results
models = {}
history_dict = {}
results = {}



for model_name, model_config in PRETRAINED_MODELS.items():
    print(f"Training model: {model_name} as fixed feature extractor.")

    # Build model with frozen base (feature extractor)
    model = build_model(model_config, freeze_base=True)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # Evaluate on validation set
    val_preds = model.predict(X_val)
    y_pred = np.argmax(val_preds, axis=1)
    y_true = np.argmax(y_val, axis=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    # Save results for reporting
    results[model_name + " (Feature Extractor)"] = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    history_dict[model_name + " (Feature Extractor)"] = history.history

    # Save model for further use (adjust file name/path as needed)
    model.save(f"{model_name}_feature_extractor.h5")

    # Now fine-tune: unfreeze the top few layers (adjust fine_tune_at index as needed)
    print(f"Fine-tuning model: {model_name}.")
    fine_tune_at = len(model_config["base_model"].layers) - 20  # unfreeze last 20 layers
    model_finetune = build_model(model_config, freeze_base=False, fine_tune_at=fine_tune_at)

    # Compile the model for fine-tuning
    model_finetune.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

    history_ft = model_finetune.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    val_preds_ft = model_finetune.predict(X_val)
    y_pred_ft = np.argmax(val_preds_ft, axis=1)
    acc_ft = accuracy_score(y_true, y_pred_ft)
    cm_ft = confusion_matrix(y_true, y_pred_ft)
    precision_ft, recall_ft, f1_ft, _ = precision_recall_fscore_support(y_true, y_pred_ft, average="weighted")

    results[model_name + " (Fine-tuned)"] = {
        "accuracy": acc_ft,
        "confusion_matrix": cm_ft,
        "precision": precision_ft,
        "recall": recall_ft,
        "f1_score": f1_ft
    }
    history_dict[model_name + " (Fine-tuned)"] = history_ft.history

    model_finetune.save(f"{model_name}_finetuned.h5")

# Plot training curves
for key, history in history_dict.items():
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label="Train")
    plt.plot(history['val_accuracy'], label="Validation")
    plt.title(f"{key} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label="Train")
    plt.plot(history['val_loss'], label="Validation")
    plt.title(f"{key} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Print evaluation results
for model_variant, metrics in results.items():
    print(f"\nModel: {model_variant}")
    print("Accuracy:", metrics["accuracy"])
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1 Score:", metrics["f1_score"])
