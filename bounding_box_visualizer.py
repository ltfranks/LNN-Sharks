import cv2
import os
import numpy as np


def draw_bounding_boxes(images_dir, predictions, output_dir, img_height, img_width, max_photos=20):
    """
    Draw bounding boxes on images based on predictions and save the output.

    Args:
        images_dir (str): Directory containing the original images.
        predictions (list): List of predicted bounding boxes and labels [(class_id, x, y, w, h), ...].
        output_dir (str): Directory to save images with bounding boxes.
        img_height (int): Image height (used for scaling bounding boxes).
        img_width (int): Image width (used for scaling bounding boxes).
        max_photos (int): Maximum number of photos to save.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(os.listdir(images_dir))[:max_photos]
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Unable to load image: {img_file}")
            continue
        image = cv2.resize(image, (img_width, img_height))

        # Draw each bounding box
        for pred in predictions[idx]:  # Predictions for the current image
            class_id, x, y, w, h = pred
            x1 = int((x - w / 2) * img_width)
            y1 = int((y - h / 2) * img_height)
            x2 = int((x + w / 2) * img_width)
            y2 = int((y + h / 2) * img_height)

            # Draw the rectangle
            color = (0, 255, 0) if class_id == 3 else (0, 0, 255)  # Green for Shark, Red for Non-Shark
            label = "Shark" if class_id == 3 else "Non-Shark"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the image
        output_path = os.path.join(output_dir, f"output_{idx + 1}.jpg")
        cv2.imwrite(output_path, image)

    print(f"{len(image_files)} images with bounding boxes saved to {output_dir}")


def generate_mock_predictions(num_images, img_height, img_width):
    """
    Generate mock predictions for testing.

    Args:
        num_images (int): Number of mock images to generate predictions for.
        img_height (int): Image height (used for scaling bounding boxes).
        img_width (int): Image width (used for scaling bounding boxes).

    Returns:
        list: Mock predictions in the format [(class_id, x, y, w, h), ...].
    """
    predictions = []
    for _ in range(num_images):
        # Mock prediction: [class_id, center_x, center_y, width, height]
        bbox = [[3, 0.5, 0.5, 0.3, 0.3]]  # Shark prediction (centered box)
        predictions.append(bbox)
    return predictions


if __name__ == "__main__":
    # Example usage
    images_dir = "test_images"
    output_dir = "output_images"
    img_height = 128
    img_width = 128

    # Generate mock predictions (replace this with real model predictions)
    predictions = generate_mock_predictions(num_images=20, img_height=img_height, img_width=img_width)

    draw_bounding_boxes(images_dir, predictions, output_dir, img_height, img_width)
