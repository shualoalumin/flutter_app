import argparse
import os
import wandb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support
import warnings
from models import create_resnet, MobileNet, GoogLeNet, base_model

class ClassMetricsCallback(Callback):
    def __init__(self, validation_data, class_names):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_true = next(self.validation_data)
        y_pred = self.model.predict(x_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_classes, 
                y_pred_classes, 
                average=None, 
                zero_division=0
            )

        print(f"\nEpoch {epoch + 1} - Class Metrics:")
        for i, (class_name, p, r, f) in enumerate(zip(self.class_names, precision, recall, f1)):
            print(f"{class_name}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")
            wandb.log({
                f'{class_name}_precision': p,
                f'{class_name}_recall': r,
                f'{class_name}_f1': f
            }, commit=False)
        
        wandb.log({}, commit=True)

def create_model(model_name, input_shape, num_classes):
    if model_name == "base_model":
        return base_model(input_shape, num_classes)
    elif model_name == "resnet":
        return create_resnet(input_shape, num_classes)
    elif model_name == "mobilenet":
        return MobileNet(input_shape, num_classes)
    elif model_name == "googlenet":
        return GoogLeNet(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def setup_data_generators(dataset_path, image_size, batch_size, valid_classes, augmentation=False):
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

    valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        classes=valid_classes
    )

    validation_generator = valid_datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        classes=valid_classes
    )

    return train_generator, validation_generator

def train_model(config, dataset_path):
    wandb.init(project="National Oceanographic AI", name=f"{config['model_name']}_{config['dataset']}", config=config)

    valid_classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d != '.ipynb_checkpoints']

    train_generator, validation_generator = setup_data_generators(
        dataset_path, 
        wandb.config.image_size, 
        wandb.config.batch_size, 
        valid_classes,
        wandb.config.augmentation
    )

    model = create_model(
        wandb.config.model_name,
        input_shape=(*wandb.config.image_size, 3),
        num_classes=wandb.config.num_classes,
        dropout_rate=wandb.config.dropout_rate
    )

    optimizer_class = getattr(tf.keras.optimizers, wandb.config.optimizer)
    optimizer = optimizer_class(learning_rate=wandb.config.learning_rate)

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        WandbMetricsLogger(log_freq="batch"),
        WandbModelCheckpoint("model_checkpoint.keras"),
        ClassMetricsCallback(validation_generator, valid_classes)
    ]

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=wandb.config.epochs,
        callbacks=callbacks
    )

    model.save(f'{wandb.config.model_name}_{wandb.config.dataset}.h5')

    artifact = wandb.Artifact(f'{wandb.config.model_name}_{wandb.config.dataset}', type='model')
    artifact.add_file(f'{wandb.config.model_name}_{wandb.config.dataset}.h5')
    wandb.log_artifact(artifact)

def main():
    parser = argparse.ArgumentParser(description='Train a CNN model')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--base_path', type=str, default='/content/National_OceanoGraphic/dataset/', help='Base path for datasets')
    parser.add_argument('--model_name', type=str, default='simple_cnn', help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use')
    parser.add_argument('--dropout_rate', type=float, default=0, help='Dropout rate')
    parser.add_argument('--augmentation', action='store_true', help='Use data augmentation')
    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "dropout_rate": args.dropout_rate,
        "image_size": (224, 224),
        "num_classes": 6,
        "augmentation": args.augmentation
    }

    dataset_path = os.path.join(args.base_path, args.dataset)
    train_model(config, dataset_path)

if __name__ == "__main__":
    main()
