import torch
from dataloader import prepare_dataloaders
from model import get_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model_path, test_loader, device):
    model = get_model('resnet18', num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    MODEL_PATH = 'tick_model.pth'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    CLASSES = ['hyalomma_female', 'hyalomma_male', 'rhipicephalus_female', 'rhipicephalus_male']
    
    _, _, test_loader = prepare_dataloaders("metadata.csv", batch_size=BATCH_SIZE)
    
    try:
        y_true, y_pred = evaluate_model(MODEL_PATH, test_loader, DEVICE)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=CLASSES))
        
        plot_confusion_matrix(y_true, y_pred, CLASSES)
        print("\nConfusion matrix plot saved as 'confusion_matrix.png'")
    except FileNotFoundError:
        print(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
