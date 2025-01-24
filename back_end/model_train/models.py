import tensorflow as tf
from tensorflow.keras import layers, models

# ResNet code
def residual_block(input_tensor, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def create_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# MobileNet code
def depthwise_separable_conv(x, filters, stride=1):
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding="same", use_bias=False)(x)
    x = layers.ReLU()(x)
    return x

def MobileNet(input_shape=(224, 224, 3), num_classes=1000):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", use_bias=False)(input_layer)
    x = layers.ReLU()(x)
    x = depthwise_separable_conv(x, 64, stride=1)
    x = depthwise_separable_conv(x, 128, stride=2)
    x = depthwise_separable_conv(x, 128, stride=1)
    x = depthwise_separable_conv(x, 256, stride=2)
    x = depthwise_separable_conv(x, 256, stride=1)
    x = depthwise_separable_conv(x, 512, stride=2)
    for _ in range(5):
        x = depthwise_separable_conv(x, 512, stride=1)
    x = depthwise_separable_conv(x, 1024, stride=2)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=input_layer, outputs=x)
    return model

# GoogLeNet code
def inception_module(x, filters):
    conv1x1 = layers.Conv2D(filters[0], (1, 1), activation="relu", padding="same")(x)
    conv1x1_3x3 = layers.Conv2D(filters[1], (1, 1), activation="relu", padding="same")(x)
    conv3x3 = layers.Conv2D(filters[2], (3, 3), activation="relu", padding="same")(conv1x1_3x3)
    conv1x1_5x5 = layers.Conv2D(filters[3], (1, 1), activation="relu", padding="same")(x)
    conv5x5 = layers.Conv2D(filters[4], (5, 5), activation="relu", padding="same")(conv1x1_5x5)
    maxpool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    conv1x1_pool = layers.Conv2D(filters[5], (1, 1), activation="relu", padding="same")(maxpool)
    return layers.concatenate([conv1x1, conv3x3, conv5x5, conv1x1_pool], axis=-1)

def GoogLeNet(input_shape=(224, 224, 3), num_classes=1000):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=2, padding="same", activation="relu")(input_layer)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    x = inception_module(x, [64, 128, 128, 32, 32, 32])
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=input_layer, outputs=x)
    return model

# Base model code
def base_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model