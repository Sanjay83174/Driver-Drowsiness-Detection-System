import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "0 FaceImages_enhanced"
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


def analyze_dataset():
    """Print a brief analysis of the dataset structure and class balance."""
    if not DATASET_DIR.exists():
        print(f"Dataset directory not found at {DATASET_DIR}")
        return

    print(f"Using dataset directory: {DATASET_DIR}")
    class_counts = {}
    total_images = 0
    for class_dir in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        count = sum(1 for f in class_dir.glob("*.jpg"))
        class_name = class_dir.name
        class_counts[class_name] = count
        total_images += count

    print("Class distribution:")
    for cls, cnt in class_counts.items():
        print(f"  - {cls}: {cnt} images")
    print(f"Total images: {total_images}")


def get_datasets():
    """Create training and validation datasets from the directory structure."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    print(f"Detected classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (
        train_ds.shuffle(1000)
        .prefetch(buffer_size=AUTOTUNE)
    )
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(input_shape=(128, 128, 3)):
    """Build a CNN model for drowsiness detection using eyes/mouth/face cues implicitly."""
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255, name="rescaling")(x)

    # Convolutional blocks with batch normalization and max pooling
    x = layers.Conv2D(32, (3, 3), padding="same", activation=None, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation=None, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation=None, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation=None, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="fc1",
    )(x)
    x = layers.Dropout(0.5, name="dropout1")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="drowsiness_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train():
    analyze_dataset()
    train_ds, val_ds, class_names = get_datasets()

    model = build_model(input_shape=IMG_SIZE + (3,))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        # Use native Keras `.keras` format instead of legacy HDF5
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "drowsiness_cnn.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks,
    )

    # Evaluate on validation set (used as held-out test in this setup)
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation (test) accuracy: {val_acc:.4f}, loss: {val_loss:.4f}")

    # Save final model as well (native Keras format)
    final_model_path = MODEL_DIR / "drowsiness_cnn_final.keras"
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")

    # Summarize best epoch metrics
    best_val_acc = max(history.history["val_accuracy"])
    best_train_acc = history.history["accuracy"][history.history["val_accuracy"].index(best_val_acc)]
    print(f"Best training accuracy: {best_train_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()

