> Note: Due to GitHub storage limitations, the dataset is not included in this repository.

This project uses an external dataset hosted on Kaggle.

## Dataset Source
- Platform: Kaggle
- Dataset Name: Lung Cancer Image Dataset
- Dataset Link: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

## Reason for Exclusion
The dataset is not included in this repository due to GitHub storage limitations and best practices for machine learning projects.

## Expected Folder Structure

After downloading and extracting the dataset, place the images as follows:

At the time of dataset download, the images are organized into four separate folders: adenocarcinoma, large.cell.carcinoma, squamous.cell.carcinoma, and normal.

For this project, the dataset is restructured for binary classification. Images in the normal folder are retained as the Normal class, 
while images from adenocarcinoma, large.cell.carcinoma, and squamous.cell.carcinoma are combined and labeled as the Cancer class.

data/
└── lung_cancer/
    ├── train/
    │   ├── normal/
    │   └── cancer/
    ├── valid/
    │   ├── normal/
    │   └── cancer/
    └── test/
        ├── normal/
        └── cancer/


## Usage
Ensure the dataset is placed in the above directory before running:
- Training scripts
- Inference scripts
- Streamlit application
