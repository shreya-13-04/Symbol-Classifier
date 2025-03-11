# Symbol-Classifier  
**Symbol Classification using Deep Learning**  

This project builds a deep learning-based symbol classifier using a CNN model to recognize and categorize symbols from images. The dataset consists of various labeled symbols, and the model is trained using PyTorch. The output is a CSV file with predicted labels for test images.  

## Project Overview  
This project implements a Convolutional Neural Network (CNN) to classify symbols from images. The model is trained on a dataset containing various symbols and predicts the corresponding labels. It leverages PyTorch for deep learning and OpenCV/PIL for image preprocessing.  

## Key Features  
- **Deep Learning Model:** A CNN-based classifier trained using PyTorch.  
- **Custom Dataset Handling:** Loads and preprocesses symbol images for training.  
- **Efficient Training:** Uses data augmentation and optimized training techniques.  
- **Evaluation & Submission:** Generates predictions and formats them in `submission.csv`.  
- **GitHub Integration:** Fully open-source with training scripts and model architecture.  

## Dataset  
The dataset consists of multiple files stored in the `/data` folder:  
- `data.zip` - Contains all symbol images. Extract to `/data`.  
- `train.csv` - Labeled dataset (image filenames + corresponding labels).  
- `test.csv` - Test dataset (image filenames only, labels hidden).  
- `metaData.csv` - Additional metadata (LaTeX representations of symbols).

 ## Model Architecture  
The CNN model consists of:  
- **Convolutional Layers** for feature extraction.  
- **ReLU Activation** to introduce non-linearity.  
- **MaxPooling Layers** for spatial reduction.  
- **Fully Connected Layers** for classification.  

## Training Process  
### Data Preprocessing  
- Resizing images to `(32, 32)`.  
- Normalization & Augmentation.  

### Training  
- **Loss Function:** Cross-Entropy Loss.  
- **Optimizer:** Adam.  
- **Learning Rate:** `0.001`.  
- **Epochs:** `10`.  

### Evaluation & Testing  
- Validating model on unseen data.  
- Generating predictions for test images.  

## Results & Performance  
The model achieves high accuracy on validation data.  
The accuracy can be further improved with data augmentation and hyperparameter tuning.  

## Future Improvements  
- Implement **Transfer Learning** for better generalization.  
- Use **Larger CNN Architectures** like ResNet.  
- Improve preprocessing with better augmentations.

**Download Dataset from Kaggle:**  
```sh
kaggle competitions download -c torch-it-up
