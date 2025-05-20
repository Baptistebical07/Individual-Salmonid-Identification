from tensorflow import keras
from PIL import Image
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import shutil
import random
from PIL import Image
tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()
from sklearn.utils.class_weight import compute_class_weight
tf.data.experimental.enable_debug_mode()
from tensorflow.keras import regularizers


name_of_the_model_FOLDER = "Baseline_ResNet_Rasp_lr_0_1"

# !!! Define model version name
name_of_the_model_version = "Baseline_ResNet_Rasp_lr_0_1_ver-1"
# !!! Set random seed for reproducibility
rnd_seed = 42

# Base output path for logs and model checkpoints
model_input_dir = Path('path/to/folder') / name_of_the_model_FOLDER
model_input_dir.mkdir(parents=True, exist_ok=True)

# Define dataset paths
output_dir = Path('path/to/folder')

# Choose training dataset type
USE_BALANCED_TRAINING = False  # Set to False to use the classic imbalanced 'train_set'
# Set hyperparameters
# batch_size = 32 
#IMAGE_SIZE = (224, 224)
PATIENCE = 50
INITIAL_LR = 0.1
MOMENTUM = 0.9
EPOCHS_PRETRAIN = 5
EPOCHS_FINE_TUNE = 80
DECAY_STEPS = 1000
DECAY_RATE = 0.96
DROPOUT = "YES"
DROPOUT_RATE = 0.5
USE_ADAM = False  # Set to True if you want to use Adam instead of SGD
WEIGHT_DECAY = 1e-4  # Try 1e-4, 5e-5, 1e-5, etc.
AUGMENTATION_STRATEGY = "strong"  # Options: "none", "light", "medium", "strong"
LR_SCHEDULE_TYPE = "exponential"  # Options: "exponential", "cosine", "constant"
USE_FOCAL_LOSS = True
LABEL_SMOOTHING = 0.0  # ONLY WORKS IF USE_FOCAL_LOSS IS SET TO False Try 0.0, 0.05, 0.1

BALANCE_SEED = 12345  # This must remain constant across versions

np.random.seed(rnd_seed)
tf.random.set_seed(rnd_seed)

train_dir = output_dir / 'Train'
valid_dir = output_dir / 'Valid'
test_dir = output_dir / 'Test'

test_sets = {
    "test_set_1": test_dir / "Test_31_05_24",
    "test_set_2": test_dir / "Test_22_01_25",
}

# Load metadata
train_data = pd.read_csv(output_dir / 'train_metadata.csv').to_dict('records')
valid_data = pd.read_csv(output_dir / 'valid_metadata.csv').to_dict('records')
test_data = {
    key: pd.read_csv(output_dir / f'{key}_metadata.csv').to_dict('records') for key in test_sets
}

# Define consistent class mapping
all_possible_classes = sorted(set(item['class'] for item in train_data))  # Ensure full 61 classes
class_mapping = {class_name: idx for idx, class_name in enumerate(all_possible_classes)}

#print("Class mapping:", class_mapping)
#print("Number of classes:", len(class_mapping))


# Load datasets
train_set = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=(224, 224), batch_size=32, seed=123)
valid_set = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir, image_size=(224, 224), batch_size=32, seed=123)
test_sets_loaded = {
    key: tf.keras.preprocessing.image_dataset_from_directory(
        test_sets[key], image_size=(224, 224), batch_size=32, seed=123) for key in test_sets
}
#print(train_set.class_names, "TRAIN SET CLASS NAMES")

# Preprocessing function
def preprocessing_function(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0  # Normalize
    #label = tf.one_hot(label, len(class_mapping))  # One-hot encode
    label = tf.one_hot(tf.clip_by_value(label, 0, len(class_mapping) - 1), len(class_mapping)) # CHANGED HERE

    return image, label

# Apply preprocessing
train_set_preprocessed = train_set.map(preprocessing_function, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
valid_set_preprocessed = valid_set.map(preprocessing_function, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_sets_preprocessed = {
    key: test_set.map(preprocessing_function, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    for key, test_set in test_sets_loaded.items()
}
#print(train_set.class_names, "TRAIN SET CLASS NAMES")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomHeight(0.2),
    tf.keras.layers.RandomWidth(0.2),
    tf.keras.layers.RandomContrast(0.1),
    #tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomTranslation(-0.1, 0.1),
    tf.keras.layers.GaussianNoise(0.01),
    tf.keras.layers.Resizing(224, 224)
])



def get_data_augmentation(strategy):
    if strategy == "none":
        return tf.keras.Sequential()
    elif strategy == "light":
        return tf.keras.Sequential([
            #tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomContrast(0.05),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.Resizing(224, 224)
        ])
    elif strategy == "medium":
        return tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomHeight(0.2),
            tf.keras.layers.RandomWidth(0.2),
            tf.keras.layers.RandomContrast(0.1),
            #tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(-0.1, 0.1),
            tf.keras.layers.GaussianNoise(0.01),
            tf.keras.layers.Resizing(224, 224)
        ])
    elif strategy == "strong":
        return tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.5),
            tf.keras.layers.RandomHeight(0.5),
            tf.keras.layers.RandomWidth(0.5),
            tf.keras.layers.RandomContrast(0.5),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(-0.3, 0.3),
            tf.keras.layers.GaussianNoise(0.05),
            tf.keras.layers.Resizing(224, 224)
        ])
    else:
        raise ValueError("Invalid augmentation strategy.")

data_augmentation = get_data_augmentation(AUGMENTATION_STRATEGY)


# Load training metadata
train_df = pd.read_csv(output_dir / 'train_metadata.csv')

# Apply  "SMOTE-inspired oversampling + augmentation" only if not already processed
if USE_BALANCED_TRAINING:
    train_metadata_balanced_path = output_dir / 'train_metadata_balanced.csv'
    balanced_images_dir = output_dir / 'balanced_images'

    # Check if SMOTE has already been applied
    if train_metadata_balanced_path.exists() and balanced_images_dir.exists() and any(balanced_images_dir.iterdir()):
        print("✅ Loading pre-balanced dataset... Skipping SMOTE application.")
        resampled_df = pd.read_csv(train_metadata_balanced_path)
    else:
        print("⚠️ Applying SMOTE to balance dataset...")
    
        # Create class-to-label mapping
        class_mapping = {cls: idx for idx, cls in enumerate(train_df['class'].unique())}
        inverse_class_mapping = {v: k for k, v
        in class_mapping.items()}
        
        #print("Class mapping:", class_mapping)
        #print("Number of classes:", len(class_mapping))

        # Convert class names to numeric labels
        train_df['label'] = train_df['class'].map(class_mapping)

        # Prepare features (X) and labels (y) for SMOTE
        y_train = train_df['label']
        
        # Find class counts
        class_counts = train_df['label'].value_counts()

        # Filter out classes with less than 2 samples
        valid_classes = class_counts[class_counts > 1].index
        filtered_df = train_df[train_df['label'].isin(valid_classes)]

        # Apply SMOTE only to valid classes
        smote = SMOTE(sampling_strategy='auto', random_state=BALANCE_SEED, k_neighbors=1)
        _, y_resampled = smote.fit_resample(np.zeros((len(filtered_df), 1)), filtered_df['label'])

        # Convert back to DataFrame
        resampled_df = pd.DataFrame({'label': y_resampled})
        resampled_df['class'] = resampled_df['label'].map(inverse_class_mapping)

        # Get original image paths for each class
        class_to_paths = train_df.groupby('class')['path'].apply(list).to_dict()

        # Function to apply augmentation and save new images
        def augment_and_save_image(original_path, save_path):
            image = Image.open(original_path).convert("RGB")  # Open image
            
            # Convert to tensor and normalize
            image_tensor = tf.keras.preprocessing.image.img_to_array(image) / 255.0
            image_tensor = tf.expand_dims(image_tensor, 0)  # Add batch dimension

            # Apply data augmentation
            augmented_tensor = data_augmentation(image_tensor, training=True)
            
            # Convert back to PIL image
            augmented_image = tf.keras.preprocessing.image.array_to_img(augmented_tensor[0])

            # Save new augmented image
            augmented_image.save(save_path)

        # Ensure reproducible image sampling
        rng_balance = random.Random(BALANCE_SEED)

        # Ensure output directory exists
        balanced_images_dir.mkdir(parents=True, exist_ok=True)

        new_paths = []
        existing_files = {file.name for file in balanced_images_dir.iterdir()}  # Track existing files

        for label in y_resampled:
            class_name = inverse_class_mapping[label]
            original_paths = class_to_paths[class_name]    

            # Select a random existing image for duplication
            original_image = rng_balance.choice(original_paths)
            new_image_name = f"{class_name}_{rng_balance.randint(10000, 99999)}.jpg"

            # Ensure unique filenames and avoid duplicate copying
            while new_image_name in existing_files:
                new_image_name = f"{class_name}_{random.randint(10000, 99999)}.jpg"

            new_image_path = balanced_images_dir / new_image_name

            augment_and_save_image(original_image, new_image_path)  # generate and save augmented images

            new_paths.append(str(new_image_path))
            existing_files.add(new_image_name)  # Add to existing file set

        # Add image paths to resampled DataFrame
        resampled_df['path'] = new_paths

        # Save balanced metadata
        resampled_df.to_csv(train_metadata_balanced_path, index=False)
        print("✅ SMOTE applied with augmentation. Augmented images saved.")

else:
    print("🚫 Skipping SMOTE and balanced training. Using original imbalanced dataset.")
    resampled_df = train_df.copy()
    resampled_df['label'] = resampled_df['class'].map(class_mapping)

# Compute class weights based on the resampled dataset
y_train_labels = resampled_df['label'].values
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

# Normalize the class weights
class_weights = class_weights / np.sum(class_weights)

# Convert to dictionary format for TensorFlow
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

#print("✅ Class weights computed and normalized:", class_weight_dict)


# Convert paths to TensorFlow dataset
image_size = (224, 224)
batch_size = 32

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Ensure all images are resized properly
    image = tf.image.resize(image, (224, 224))
    image.set_shape((224, 224, 3))  # Explicitly set shape
    
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, len(class_mapping))
    
    return image, label




train_df = pd.read_csv(output_dir / 'train_metadata.csv')
valid_df = pd.read_csv(output_dir / 'valid_metadata.csv')

# Step 1: Identify underrepresented validation classes
valid_df['label'] = valid_df['class'].map(class_mapping)
valid_class_counts = valid_df['label'].value_counts()
median_valid_count = valid_class_counts.median()
minority_classes = valid_class_counts[valid_class_counts < median_valid_count].index.tolist()

#print("⚖️ Minority classes in validation set:", minority_classes)


from tqdm import tqdm

# Augment function (lighter)
def light_augment_and_save(path, save_path):
    image = Image.open(path).convert("RGB")
    image_tensor = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    image_tensor = tf.expand_dims(image_tensor, 0)
    
    aug = tf.keras.Sequential([
        #tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomContrast(0.05),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ])
    
    augmented = aug(image_tensor, training=True)
    augmented_image = tf.keras.preprocessing.image.array_to_img(augmented[0])
    augmented_image.save(save_path)

# Directory to store augmented validation images
valid_augmented_dir = output_dir / 'Valid_augmented'
valid_augmented_dir.mkdir(parents=True, exist_ok=True)

augmented_paths = []
augmented_labels = []

for label in tqdm(minority_classes):
    class_name = [k for k, v in class_mapping.items() if v == label][0]
    paths = valid_df[valid_df['label'] == label]['path'].tolist()
    augment_factor = int(median_valid_count // len(paths))  # How many more per original
    
    rng_val_aug = random.Random(BALANCE_SEED) # Set the constant seed for validation augmentation dataset

    for path in paths:
        for _ in range(augment_factor):
            new_name = f"{class_name}_{rng_val_aug.randint(10000,99999)}.jpg"
            save_path = valid_augmented_dir / new_name
            light_augment_and_save(path, save_path)

            augmented_paths.append(str(save_path))
            augmented_labels.append(label)

# Combine original and augmented validation data
final_valid_paths = valid_df['path'].tolist() + augmented_paths
final_valid_labels = valid_df['label'].tolist() + augmented_labels

valid_dataset = tf.data.Dataset.from_tensor_slices((final_valid_paths, final_valid_labels))
valid_dataset = valid_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

#print(f"✅ Validation set augmented. New size: {len(final_valid_paths)}")


# Log directory for TensorBoard
log_dir = model_input_dir / "logs" / f"{name_of_the_model_version}_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
log_dir.mkdir(parents=True, exist_ok=True)

# Best model save path
best_model_save_path = model_input_dir / f"{name_of_the_model_version}.keras"

# Define callbacks
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    best_model_save_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1
)
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# Create TF dataset
train_paths = resampled_df['path'].values
train_labels = resampled_df['label'].values

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

@tf.function
def augment(image, label):
    image = data_augmentation(image, training=True)
    image = tf.image.resize(image, (224, 224))  # Ensure consistent size
    return image, label


# Modify dataset mapping
train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(5000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)



# Load ResNet50 model
base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# OPTIONAL: Apply L2 regularization to base_model layers
for layer in base_model.layers:
    if hasattr(layer, 'kernel_regularizer'):
        layer.kernel_regularizer = regularizers.l2(WEIGHT_DECAY)  # Add L2 regularization

# Add custom head
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

if DROPOUT == "YES":
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

# 🔥 Dense layer with L2 regularization
output = tf.keras.layers.Dense(len(class_mapping), 
                               activation="softmax", 
                               kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

def get_lr_schedule(schedule_type):
    if schedule_type == "exponential":
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=INITIAL_LR,
            decay_steps=DECAY_STEPS,
            decay_rate=DECAY_RATE,
            staircase=True
        )
    elif schedule_type == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=INITIAL_LR,
            decay_steps=EPOCHS_FINE_TUNE * len(train_dataset)
        )
    elif schedule_type == "constant":
        return INITIAL_LR
    else:
        raise ValueError("Invalid learning rate schedule.")

lr_schedule = get_lr_schedule(LR_SCHEDULE_TYPE)


# Choose optimizer based on condition
if USE_ADAM:
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
else:
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=MOMENTUM)


# Custom Focal Loss (multi-class version)
def focal_loss(gamma=2., alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)
    return loss_fn

loss_fn = focal_loss() if USE_FOCAL_LOSS else "categorical_crossentropy"
# If not using focal loss:
if not USE_FOCAL_LOSS:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)



# Pretraining: Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=["accuracy"])


# Choose training dataset
final_train_set = train_dataset if USE_BALANCED_TRAINING else train_set_preprocessed

# Choose class weights accordingly
final_class_weight = class_weight_dict if USE_BALANCED_TRAINING else None

# Train the model with history tracking
model.fit(final_train_set, epochs=EPOCHS_PRETRAIN, validation_data=valid_set_preprocessed)


# Fine-tune: Unfreeze base layers
for layer in base_model.layers:
    layer.trainable = True

# ⚠️ Create a **new optimizer** after changing the trainable variables
optimizer_finetune = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=MOMENTUM)

model.compile(loss=loss_fn,
              optimizer=optimizer_finetune,
              metrics=["accuracy"])


# Choose training dataset
final_train_set = train_dataset if USE_BALANCED_TRAINING else train_set_preprocessed

# Choose class weights accordingly
final_class_weight = class_weight_dict if USE_BALANCED_TRAINING else None

# Train the model with history tracking
history = model.fit(
    final_train_set,
    epochs=EPOCHS_FINE_TUNE,
    validation_data=valid_set_preprocessed,
    class_weight=final_class_weight,
    callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb]
)

