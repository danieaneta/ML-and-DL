from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision import transforms
import torch

def get_cityscapes_loaders(data_dir, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    train_dataset = Cityscapes(
        root=data_dir,
        split='train',
        mode='fine',
        target_type='semantic',
        transform=transform,
        target_transform=mask_transform
    )
    
    val_dataset = Cityscapes(
        root=data_dir,
        split='val',
        mode='fine',
        target_type='semantic',
        transform=transform,
        target_transform=mask_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader