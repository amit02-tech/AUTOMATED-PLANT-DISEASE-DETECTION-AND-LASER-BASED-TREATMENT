import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# --- PART 1: HARDWARE & PATH ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ RTX 3050 GPU Activated")
    except RuntimeError as e: print(e)

DATASET_DIR = 'dataset' # Ensure your 8 class folders are inside here

# --- PART 2: DATA AUGMENTATION ---
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
    DATASET_DIR, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='validation'
)

# --- PART 3: MODEL ARCHITECTURE ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- PART 4: SMART CALLBACKS ---
# 1. Saves ONLY the model with the highest validation accuracy
checkpoint = ModelCheckpoint('best_plant_disease_model.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             verbose=1)

# 2. Stops training if the model stops improving for 5 epochs
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               restore_best_weights=True,
                               verbose=1)

# 3. Slows down learning if it gets stuck
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=3,
                              min_lr=1e-6,
                              verbose=1)

callbacks_list = [checkpoint, early_stopping, reduce_lr]

# --- PART 5: PHASE 1 - WARM UP ---
print("\n--- PHASE 1: WARM UP ---")
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator,
          validation_data=validation_generator,
          epochs=10,
          callbacks=[checkpoint]) # Just save the best one here

# --- PART 6: PHASE 2 - DEEP FINE TUNING ---
print("\n--- PHASE 2: DEEP FINE TUNING ---")
base_model.trainable = True

# Unfreeze the last 30 layers for the GPU to chew on
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Run for up to 30 epochs, but Early Stopping will likely stop it sooner when it peaks
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=30,
                    callbacks=callbacks_list)

print("\n✅ Training Complete. Best model saved as 'best_plant_disease_model.h5'")