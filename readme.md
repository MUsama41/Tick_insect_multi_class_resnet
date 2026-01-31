# Tick Classification Project

A deep learning project to classify different species and genders of ticks using PyTorch and Transfer Learning.

## 🛠️ Setup and Installation

1.  **Clone the repository** (or navigate to the project folder).
2.  **Create a virtual environment**:
    ```powershell
    python -m venv venv
    ```
3.  **Activate the environment**:
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
4.  **Install dependencies**:
    ```powershell
    pip install torch torchvision torchaudio pandas scikit-learn matplotlib seaborn pillow
    ```

## 🧠 Model Architecture
-   **Base Model**: ResNet18 (Pre-trained on ImageNet).
-   **Customization**: The final fully connected layer was replaced to match our 4 target classes.
-   **Optimizer**: Adam with a learning rate of 0.001.
-   **Loss Function**: Cross-Entropy Loss.

## 📊 Dataset Structure
The dataset is split into 4 categories:
1.  **hyalomma_female** (Label Index: 0)
2.  **hyalomma_male** (Label Index: 1)
3.  **rhipicephalus_female** (Label Index: 2)
4.  **rhipicephalus_male** (Label Index: 3)

### Data Split
The project uses a stratified split to ensure equal representation of classes:
-   **Training**: 70%
-   **Validation**: 15% (Used for saving the "Best Model")
-   **Testing**: 15% (Used for final evaluation)

## 🚀 How to Run

### 1. Data Preparation
Generate the `metadata.csv` file from your raw images:
```powershell
python generate_metadata.py
```

### 2. Training
Train the model. The script saves the best weights as `tick_model.pth` based on validation accuracy:
```powershell
python train.py
```

### 3. Evaluation
Run evaluation on the test set to see precision, recall, and F1-scores:
```powershell
python evaluate.py
```

### 4. Inference
To predict a single image, use the Jupyter Notebook:
-   Open `inference.ipynb`
-   Point `sample_image` to your image path and run all cells.

## 📂 Project Structure
-   `model.py`: Architecture definition.
-   `dataloader.py`: Image loading and augmentation pipeline.
-   `train.py`: Training and validation loops.
-   `evaluate.py`: Performance metrics and confusion matrix generation.
-   `inference.ipynb`: Simple pipeline for single-image prediction.
-   `generate_metadata.py`: Helper script to organize dataset paths.
