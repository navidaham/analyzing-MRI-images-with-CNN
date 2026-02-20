# analyzing-MRI-images-with-CNN

Description: This repository hosts Project 3 for the course FYS-STK4155 (Applied Data Analysis and Machine Learning). The project addresses the critical task of automatic brain tumor classification from Magnetic Resonance Imaging (MRI) scans where the brain tumors are in 4 categories: no tumor, meningioma, pituitary tumor and glioma.

The methodology involves developing and comparing two distinct machine learning architectures:

A Convolutional Neural Network (CNN) for robust feature extraction directly from the image data.

A standard Artificial Neural Network (ANN), using the flattened images, to serve as a performance baseline.

The primary goal is to evaluate and contrast the performance (accuracy and F1) of the CNN against the ANN for this specific classification task, providing insight into the efficacy of deep learning techniques for medical image analysis.

The data is openly available at: https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet/tree/master Download and add to the ANN folder.

### Setup Instructions

1. Clone the repository:
   ```
   git clone git@github.com/navidaham/analyzing-MRI-images-with-CNN.git

   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt

### Running the code

For ANN: run `run_full_grid_and_heatmaps.py`, choose best parameters and put in `run_final_ann.py`, update parameters in `evaluate_ann.py` and run.
For CNN: run `CNN_and_data_preprocessing.ipynb`.
