import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    'rice_image_dataset',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    'rice_image_dataset',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation'
)

base_model = MobileNetV2(input_shape=(img_size, img_size, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=10)
model.save("rice_model.h5")
print("Model saved as rice_model.h5")