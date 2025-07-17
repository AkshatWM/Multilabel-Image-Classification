from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Loading the dataset
dataset, info = tfds.load('voc/2007', with_info=True, split=['train', 'validation'])
train_ds, test_ds = dataset

#Stucture of the data
sample_element = next(iter(train_ds))
print("Keys in sample element:", sample_element.keys())

def print_structure(element, indent=0):
    for key, value in element.items():
        if isinstance(value, dict):
            print(f"{'  ' * indent}Key: {key}, Data type: dict")
            print_structure(value, indent + 1)
        else:
            print(f"{'  ' * indent}Key: {key}, Data type: {value.dtype}, Shape: {value.shape}")

print_structure(sample_element)

#Preprocessing function
def preprocess_image(element):
    image = element['image']
    labels = element['labels']

    # Resize
    image = tf.image.resize(image, [224, 224])

    # Apply data augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_crop(image, size=[224, 224, 3])

    #normalize
    image = tf.cast(image, tf.float32) / 255.0

    # Prepare multilabel vector
    label_vector = tf.zeros(20, dtype=tf.int64)
    label_vector = tf.tensor_scatter_nd_update(
        label_vector,
        tf.expand_dims(labels, axis=1),
        tf.ones_like(labels, dtype=tf.int64)
    )

    return image, label_vector

#Applying the preprocessing to the dataset
train_ds_preprocessed = train_ds.map(preprocess_image)
test_ds_preprocessed = test_ds.map(preprocess_image)

batch_size = 16

train_ds_batched = train_ds_preprocessed.batch(batch_size)
test_ds_batched = test_ds_preprocessed.batch(batch_size)

train_ds_prefetched = train_ds_batched.prefetch(tf.data.AUTOTUNE)
test_ds_prefetched = test_ds_batched.prefetch(tf.data.AUTOTUNE)

#Preparing the model
base_model = tf.keras.applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg')

base_model.trainable=True
for i in base_model.layers[:-10]:
    i.trainable=False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(20, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adamW',
     metrics=[
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.2),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')]
)
model.summary()

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.weights.h5',
    monitor='val_recall',
    mode='max',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_recall',
    mode='max',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

#Training the model
history = model.fit(train_ds_prefetched, validation_data=test_ds_prefetched, epochs=15, verbose=1, callbacks=[checkpoint_cb, earlystop_cb])

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
