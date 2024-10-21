from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import torch

def get_voc_loaders(data_dir, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.where(x > 0, 1, 0))  # Binary mask
    ])
    
    train_dataset = VOCSegmentation(
        root=data_dir,
        year='2012',
        image_set='train',
        download=True,
        transform=transform,
        target_transform=mask_transform
    )
    
    val_dataset = VOCSegmentation(
        root=data_dir,
        year='2012',
        image_set='val',
        download=True,
        transform=transform,
        target_transform=mask_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader