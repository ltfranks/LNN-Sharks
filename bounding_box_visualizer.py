# bounding_box_visualizer.py

import cv2
import os

def draw_ground_truth_bounding_boxes(images_dir, labels_dir, output_dir, img_height, img_width, max_photos=20):
    """
    Draw ground truth bounding boxes on images based on YOLO label files.

    Args:
        images_dir (str): Directory containing the original images.
        labels_dir (str): Directory containing the YOLO label files.
        output_dir (str): Directory to save images with bounding boxes.
        img_height (int): Image height (used for resizing).
        img_width (int): Image width (used for resizing).
        max_photos (int): Maximum number of photos to process.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(os.listdir(images_dir))[:max_photos]
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Unable to load image: {img_file}")
            continue
        image = cv2.resize(image, (img_width, img_height))

        if not os.path.exists(label_path):
            print(f"No label file for image: {img_file}")
            continue

        # Read label file and draw bounding boxes
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert normalized coordinates to pixel values
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    # Draw the rectangle
                    color = (0, 255, 0) if class_id == 3 else (0, 0, 255)
                    label = "Shark" if class_id == 3 else f"Class {class_id}"
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the image
        output_path = os.path.join(output_dir, f"output_{idx + 1}.jpg")
        cv2.imwrite(output_path, image)

    print(f"{len(image_files)} images with ground truth bounding boxes saved to {output_dir}")
