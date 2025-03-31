# LNN-Sharks

![Capture](https://github.com/user-attachments/assets/0787296b-605c-470d-8d40-5bcec76cd4d5)

LNN-Sharks is a project focused on leveraging Liquid Time-Constant Networks (LTCs) to analyze and predict patterns in shark imagery. The repository contains code and resources for processing the SharkSpotting dataset.

## Dataset

The SharkSpotting dataset is utilized for training and evaluating models. Key features include:

- **Total Images**: 6,953 images.
- **Median Image Dimensions**: 640x640.
- **Dataset Splits**: Organized into train, validation, and test directories.
- **Image Types**: JPG.
- **Annotations**: YOLOv9 format annotations are provided for each image.

Access the dataset here: [SharkSpotting Dataset](https://universe.roboflow.com/sharkspotting-uwbou/sharkspotting-nixfq/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true).

## Repository Structure

- `main.py`: Entry point for training and evaluation scripts.
- `bounding_box_visualizer.py`: Utility for visualizing bounding boxes on images.
- `output_images/`: Directory containing generated images.
- `output_video.mp4`: Video compilation of processed images.
- `shark_confusion_matrix.png`: Confusion matrix of model predictions.
- `.idea/`: Project configuration files (specific to JetBrains IDEs).

## Requirements

- Python 3.x
- Required Python packages are listed in `requirements.txt`. Install them using:

  ```bash
  pip install -r requirements.txt

