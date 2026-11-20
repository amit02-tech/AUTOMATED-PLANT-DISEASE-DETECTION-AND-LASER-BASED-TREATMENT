import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATASET_DIR = 'dataset'
MODEL_SAVE_NAME = 'best_mango_disease_model.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
WARMUP_EPOCHS = 10
TUNE_EPOCHS = 30
WARMUP_LR = 0.0005
TUNE_LR = 0.00001

# --- HARDWARE SETUP ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ RTX 3050 GPU Memory Growth Configured")
    except RuntimeError as e:
        print(f"GPU Error: {e}")

# --- DATA PIPELINE ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation'
)

# --- ARCHITECTURE ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# --- CALLBACKS ---
callbacks_list = [
    ModelCheckpoint(MODEL_SAVE_NAME, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# --- PHASE 1: WARM UP ---
print("\n--- PHASE 1: WARM UP ---")
model.compile(optimizer=Adam(learning_rate=WARMUP_LR), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=WARMUP_EPOCHS, callbacks=[callbacks_list[0]])

# --- PHASE 2: DEEP FINE TUNING ---
print("\n--- PHASE 2: DEEP FINE TUNING ---")
base_model.trainable = True

# Unfreeze the last 30 layers for the GPU to chew on
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=TUNE_LR), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=validation_generator, epochs=TUNE_EPOCHS, callbacks=callbacks_list)

# --- METRIC LOGGING ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.savefig('training_history.png')
print(f"\n✅ Training Complete. Best model saved as: {MODEL_SAVE_NAME}")