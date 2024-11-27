import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def initialize_weights(input_dim, reservoir_dim, output_dim, spectral_radius):
    # Reservoir weights initialized randomly
    reservoir_weights = np.random.randn(reservoir_dim, reservoir_dim)
    # Scale reservoir weights to achieve desired spectral radius
    reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(reservoir_weights)))
    input_weights = np.random.randn(reservoir_dim, input_dim)

    # Initialize output weights to small random values
    output_weights = np.random.randn(reservoir_dim, output_dim) * 0.01

    return reservoir_weights, input_weights, output_weights


def train_lnn(batch_input, batch_labels, reservoir_weights, input_weights, output_weights,
              leak_rate, reg_lambda, learning_rate):
    """
    Train the LNN on a single batch of data.
    """
    batch_size, reservoir_dim = len(batch_input), reservoir_weights.shape[0]

    # Convert weights to TensorFlow tensors
    reservoir_weights_tf = tf.convert_to_tensor(reservoir_weights, dtype=tf.float32)
    input_weights_tf = tf.convert_to_tensor(input_weights, dtype=tf.float32)
    output_weights_tf = tf.Variable(output_weights, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    reservoir_states = np.zeros((batch_size, reservoir_dim))

    for i in range(batch_size):
        # Update reservoir state
        if i > 0:
            reservoir_states[i, :] = (1 - leak_rate) * reservoir_states[i - 1, :]
        reservoir_states[i, :] += leak_rate * np.tanh(
            np.dot(input_weights_tf.numpy(), batch_input[i, :]) +
            np.dot(reservoir_weights_tf.numpy(), reservoir_states[i, :])
        )

    reservoir_states_tf = tf.convert_to_tensor(reservoir_states, dtype=tf.float32)
    batch_labels_tf = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

    # GradientTape for training
    with tf.GradientTape() as tape:
        logits = tf.matmul(reservoir_states_tf, output_weights_tf)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=batch_labels_tf, logits=logits)
        ) + reg_lambda * tf.reduce_sum(tf.square(output_weights_tf))

    gradients = tape.gradient(loss, [output_weights_tf])
    optimizer.apply_gradients(zip(gradients, [output_weights_tf]))

    train_predictions = np.dot(reservoir_states, output_weights_tf.numpy())
    train_accuracy = np.mean(
        np.argmax(train_predictions, axis=1) == np.argmax(batch_labels, axis=1)
    )
    return output_weights_tf.numpy(), train_accuracy


def predict_lnn(input_data, reservoir_weights, input_weights, output_weights, leak_rate):
    num_samples = input_data.shape[0]
    reservoir_dim = reservoir_weights.shape[0]
    reservoir_states = np.zeros((num_samples, reservoir_dim))

    for i in range(num_samples):
        # Update reservoir state
        if i > 0:
            reservoir_states[i, :] = (1 - leak_rate) * reservoir_states[i - 1, :]
        reservoir_states[i, :] += leak_rate * np.tanh(
            np.dot(input_weights, input_data[i, :]) +
            np.dot(reservoir_weights, reservoir_states[i, :])
        )

    # Compute predictions
    predictions = np.dot(reservoir_states, output_weights)
    return predictions


class ImageDataGenerator(Sequence):
    def __init__(self, images_dir, labels_dir, img_height, img_width, batch_size, num_classes):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_files = sorted(os.listdir(images_dir))
        self.indexes = np.arange(len(self.image_files))

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))  # Use ceiling to include all samples

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_files = [self.image_files[k] for k in batch_indexes]
        images, labels = [], []

        for img_file in batch_files:
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Load and preprocess image
            img_path = os.path.join(self.images_dir, img_file)
            img = load_img(img_path, target_size=(self.img_height, self.img_width))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)

            # Process label
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_file)
            image_contains_shark = 0  # 0: Non-Shark, 1: Shark
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5 and int(parts[0]) == 3:  # Shark class ID
                            image_contains_shark = 1
                            break
            labels.append(image_contains_shark)

        images = np.array(images)
        labels = keras.utils.to_categorical(labels, num_classes=self.num_classes)
        return images, labels

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if desired
        np.random.shuffle(self.indexes)


def plot_confusion_matrix(all_labels, all_predictions, class_names):
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def main():
    # Set image dimensions and batch size
    img_height = 128
    img_width = 128
    num_channels = 3
    batch_size = 64
    num_classes = 2

    # Paths to your dataset directories
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    train_images_dir = os.path.join(data_dir, 'train', 'images')
    train_labels_dir = os.path.join(data_dir, 'train', 'labels')
    test_images_dir = os.path.join(data_dir, 'test', 'images')
    test_labels_dir = os.path.join(data_dir, 'test', 'labels')

    # Create generators
    train_generator = ImageDataGenerator(train_images_dir, train_labels_dir, img_height, img_width, batch_size,
                                         num_classes)
    test_generator = ImageDataGenerator(test_images_dir, test_labels_dir, img_height, img_width, batch_size,
                                        num_classes)

    # Calculate steps per epoch dynamically
    steps_per_epoch = len(train_generator)  # Number of batches in the training data
    validation_steps = len(test_generator)  # Number of batches in the test data

    # Build a pretrained CNN model for feature extraction
    cnn_model = keras.models.Sequential([
        keras.applications.ResNet50(weights='imagenet', include_top=False,
                                    input_shape=(img_height, img_width, num_channels)),
        keras.layers.GlobalAveragePooling2D()
    ])

    # Determine `input_dim` dynamically
    for x_train_batch, _ in train_generator:
        try:
            x_train_features_batch = cnn_model.predict(x_train_batch, verbose=0)
            input_dim = x_train_features_batch.shape[1]
            break  # Only process one batch to determine `input_dim`
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            continue

    # Initialize LNN weights
    reservoir_dim = 256
    output_dim = num_classes
    spectral_radius = 0.8
    leak_rate = 0.2
    num_epochs = 10
    reg_lambda = 1e-4
    learning_rate = 1e-2

    reservoir_weights, input_weights, output_weights = initialize_weights(
        input_dim, reservoir_dim, output_dim, spectral_radius
    )

    # Train the LNN
    print("Training LNN...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_accuracy = []
        batch_count = 0  # Track the number of processed batches

        for x_train_batch, y_train_batch in train_generator:
            try:
                x_train_features_batch = cnn_model.predict(x_train_batch, verbose=0)

                # Train LNN on the batch
                output_weights, batch_accuracy = train_lnn(
                    x_train_features_batch, y_train_batch, reservoir_weights, input_weights, output_weights,
                    leak_rate, reg_lambda, learning_rate
                )
                epoch_accuracy.append(batch_accuracy)
                batch_count += 1

                # Stop processing if steps_per_epoch is reached
                if batch_count >= steps_per_epoch:
                    break
            except Exception as e:
                print(f"Error during training batch: {e}")
                continue

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {np.mean(epoch_accuracy):.4f}")

        # Evaluate the model
    print("Evaluating on test data...")
    all_labels = []
    all_predictions = []

    batch_count = 0  # Track processed batches
    for x_test_batch, y_test_batch in test_generator:
        try:
            print(f"Evaluating batch {batch_count + 1}...")
            x_test_features_batch = cnn_model.predict(x_test_batch, verbose=0)

            # Predict with LNN
            test_predictions = predict_lnn(
                x_test_features_batch, reservoir_weights, input_weights, output_weights, leak_rate
            )
            all_predictions.extend(np.argmax(test_predictions, axis=1))
            all_labels.extend(np.argmax(y_test_batch, axis=1))
            batch_count += 1

            if batch_count >= validation_steps:
                break
        except Exception as e:
            print(f"Error during evaluation batch: {e}")
            continue

    # Check if any predictions were made
    if not all_predictions:
        print("No predictions were made during evaluation. Check your test data or generator.")
        return

    # Classification report
    print(classification_report(all_labels, all_predictions, target_names=['Non-Shark', 'Shark']))

    # Confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, ['Non-Shark', 'Shark'])


if __name__ == "__main__":
    main()
