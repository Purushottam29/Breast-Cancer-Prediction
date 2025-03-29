import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load Pretrained DenseNet121 (without top layer)
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom classifier layers
x = GlobalAveragePooling2D()(base_model.output)  # Use GAP instead of Flatten
x = Dense(128, activation="relu")(x)
output = Dense(3, activation="softmax")(x)  # Properly assigned output

# Create final model
model = Model(inputs=base_model.input, outputs=output)  # Use "output" instead of "x"

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Prepare data (replace with your dataset path)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(
    "./dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",  # Change to "categorical" for multi-class classification
    subset="training"
)

val_data = datagen.flow_from_directory(
    "./dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Train model
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("model/densenet121_model.h5")
