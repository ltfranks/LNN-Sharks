import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

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

    Args:
        batch_input: Features for the current batch.
        batch_labels: Labels for the current batch.
        reservoir_weights: Reservoir weight matrix.
        input_weights: Input weight matrix.
        output_weights: Output weight matrix (to be updated).
        leak_rate: Leak rate for reservoir state update.
        reg_lambda: Regularization parameter.
        learning_rate: Learning rate for the optimizer.

    Returns:
        Updated output_weights.
    """
    batch_size, reservoir_dim = len(batch_input), reservoir_weights.shape[0]

    # Convert weights to TensorFlow tensors
    reservoir_weights_tf = tf.convert_to_tensor(reservoir_weights, dtype=tf.float32)
    input_weights_tf = tf.convert_to_tensor(input_weights, dtype=tf.float32)
    output_weights_tf = tf.Variable(output_weights, dtype=tf.float32)

    # Use Adam optimizer to adjust weights
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Initialize reservoir state
    reservoir_states = np.zeros((batch_size, reservoir_dim))

    for i in range(batch_size):
        # Update reservoir state
        if i > 0:
            reservoir_states[i, :] = (1 - leak_rate) * reservoir_states[i - 1, :]
        reservoir_states[i, :] += leak_rate * np.tanh(
            np.dot(input_weights_tf.numpy(), batch_input[i, :]) +
            np.dot(reservoir_weights_tf.numpy(), reservoir_states[i, :])
        )

    # Convert to TensorFlow tensors
    reservoir_states_tf = tf.convert_to_tensor(reservoir_states, dtype=tf.float32)
    batch_labels_tf = tf.convert_to_tensor(batch_labels, dtype=tf.float32)

    # GradientTape for training
    with tf.GradientTape() as tape:
        logits = tf.matmul(reservoir_states_tf, output_weights_tf)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=batch_labels_tf, logits=logits)
        ) + reg_lambda * tf.reduce_sum(tf.square(output_weights_tf))

    # Update output weights
    gradients = tape.gradient(loss, [output_weights_tf])
    #print(f"Gradient Norm: {np.linalg.norm([g.numpy() for g in gradients])}")
    optimizer.apply_gradients(zip(gradients, [output_weights_tf]))

    # Compute training accuracy for this batch
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

def load_and_preprocess_data(images_dir, labels_dir, img_height, img_width, batch_size=32, num_classes=2):
    """
    Loads images and labels in batches using a generator.

    Args:
        images_dir (str): Path to the directory containing images.
        labels_dir (str): Path to the directory containing labels.
        img_height (int): The height to resize images to.
        img_width (int): The width to resize images to.
        batch_size (int): Number of samples per batch.
        num_classes (int): Number of classes for one-hot encoding.

    Returns:
        Generator yielding batches of images and labels.
    """
    image_files = sorted(os.listdir(images_dir))

    def generator():
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            images = []
            labels = []
            for img_file in batch_files:
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                # Load and preprocess image
                img_path = os.path.join(images_dir, img_file)
                img = load_img(img_path, target_size=(img_height, img_width))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)

                # Process label
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)
                image_contains_shark = False
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5 and int(parts[0]) == 3:  # Shark class ID
                                image_contains_shark = True
                                break

                labels.append(0 if image_contains_shark else 1)

            # Convert to arrays and one-hot encode labels
            images = np.array(images)
            labels = keras.utils.to_categorical(labels, num_classes=num_classes)

            yield images, labels

    return generator()


def main():
    # Set image dimensions and batch size
    img_height = 128
    img_width = 128
    num_channels = 3
    batch_size = 64
    # Paths to your dataset directories
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    train_images_dir = os.path.join(data_dir, 'train', 'images')
    train_labels_dir = os.path.join(data_dir, 'train', 'labels')
    test_images_dir = os.path.join(data_dir, 'test', 'images')
    test_labels_dir = os.path.join(data_dir, 'test', 'labels')

    # Build a CNN model for feature extraction
    cnn_model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, num_channels)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten()
    ])

    # Determine `input_dim` dynamically
    train_generator = load_and_preprocess_data(train_images_dir, train_labels_dir, img_height, img_width, batch_size)
    for x_train_batch, _ in train_generator:
        x_train_features_batch = cnn_model.predict(x_train_batch)
        input_dim = x_train_features_batch.shape[1]
        break  # Only process one batch to determine `input_dim`

    # Initialize LNN weights
    reservoir_dim = 256
    output_dim = 2
    spectral_radius = 0.8
    leak_rate = 0.2
    num_epochs = 10
    reg_lambda = 1e-4
    learning_rate = 1e-2

    reservoir_weights, input_weights, output_weights = initialize_weights(
        input_dim, reservoir_dim, output_dim, spectral_radius
    )

    # Train LNN
    print("Training LNN...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_accuracy = []
        train_generator = load_and_preprocess_data(train_images_dir, train_labels_dir, img_height, img_width,
                                                   batch_size)
        for x_train_batch, y_train_batch in train_generator:
            x_train_features_batch = cnn_model.predict(x_train_batch, verbose=0)

            # Train LNN on the batch
            output_weights, batch_accuracy = train_lnn(
                x_train_features_batch, y_train_batch, reservoir_weights, input_weights, output_weights,
                leak_rate, reg_lambda, learning_rate
            )
            epoch_accuracy.append(batch_accuracy)

        # Print average accuracy for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {np.mean(epoch_accuracy):.4f}")

    # Evaluate test data
    print("Evaluating on test data in batches...")
    test_generator = load_and_preprocess_data(test_images_dir, test_labels_dir, img_height, img_width, batch_size)
    test_accuracy_total = 0
    num_batches = 0

    for x_test_batch, y_test_batch in test_generator:
        x_test_features_batch = cnn_model.predict(x_test_batch)
        test_predictions = predict_lnn(
            x_test_features_batch, reservoir_weights, input_weights, output_weights, leak_rate
        )
        batch_accuracy = np.mean(
            np.argmax(test_predictions, axis=1) == np.argmax(y_test_batch, axis=1)
        )
        test_accuracy_total += batch_accuracy
        num_batches += 1

    print(f"Test Accuracy: {test_accuracy_total / num_batches:.4f}")

if __name__ == "__main__":
    main()


