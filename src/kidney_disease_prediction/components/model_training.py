import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from kidney_disease_prediction.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        base_model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        # Add resizing layer INSIDE the function
        inputs = tf.keras.Input(shape=(None, None, 3))
        x = tf.keras.layers.Resizing(
            self.config.params_image_size[0],
            self.config.params_image_size[1]
        )(inputs)
        x = base_model(x)

        self.model = tf.keras.Model(inputs, x)

        # 🔥 compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("Base model loaded, resized, and compiled.")

    # -----------------------------------------------------
    # Data Pipeline
    # -----------------------------------------------------
    def train_valid_generator(self):

        IMG_SIZE = tuple(self.config.params_image_size[:-1])
        BATCH = self.config.params_batch_size

        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="training",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH
        )

        valid_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="validation",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=BATCH
        )

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)

        # Optional augmentation
        if self.config.params_is_augmentation:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1)
            ])

            train_ds = train_ds.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE
            )

        self.train_dataset = train_ds
        self.valid_dataset = valid_ds

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        print(f"Model saved at: {path}")

    def train(self):
        self.model.fit(
            self.train_dataset,
            epochs=self.config.params_epochs,
            validation_data=self.valid_dataset
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
