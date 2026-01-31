import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import prepare_dataloaders
from model import get_model
import time
import copy

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device='cpu', save_path='tick_model.pth'):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, dataloaders['val'], criterion, device)
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            
        print()
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Data
    train_loader, val_loader, _ = prepare_dataloaders("metadata.csv", batch_size=BATCH_SIZE)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # Model
    model = get_model('resnet18', num_classes=4).to(DEVICE)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE)
