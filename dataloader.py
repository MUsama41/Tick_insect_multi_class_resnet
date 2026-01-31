import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

class TickDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.classes = sorted(self.dataframe['label'].unique())
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['file_path']
        label_idx = self.dataframe.iloc[idx]['label_idx']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def prepare_dataloaders(csv_path, batch_size=32):
    df = pd.read_csv(csv_path)
    
    # Split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label_idx'])
    # Split Temp: 50% Val, 50% Test (results in 15% each of total)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label_idx'])
    
    train_trans, val_trans = get_transforms()
    
    train_dataset = TickDataset(train_df, transform=train_trans)
    val_dataset = TickDataset(val_df, transform=val_trans)
    test_dataset = TickDataset(test_df, transform=val_trans)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_one_hot(label_idx, num_classes=4):
    """
    Utility for one-hot encoding if needed elsewhere.
    """
    return np.eye(num_classes)[label_idx]

if __name__ == "__main__":
    # Test dataloader
    t_loader, v_loader, ts_loader = prepare_dataloaders("metadata.csv")
    img, lbl = next(iter(t_loader))
    print(f"Batch image shape: {img.shape}")
    print(f"Batch label shape: {lbl.shape}")
    print(f"Sample one-hot: {get_one_hot(lbl[0].item())}")
