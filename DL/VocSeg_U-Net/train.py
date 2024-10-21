import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import UNet
from utils.data_loader import get_voc_loaders
from tqdm import tqdm
import logging
import os
from config import Config

def train():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device(Config.DEVICE)
    logger.info(f'Using device: {device}')

    model = UNet(in_channels=Config.IN_CHANNELS, out_channels=Config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    train_loader, val_loader = get_voc_loaders(Config.DATA_DIR, Config.BATCH_SIZE)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS}')
        
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        logger.info(f'Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {epoch_loss:.4f}')
        
        if (epoch + 1) % Config.VAL_FREQUENCY == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.squeeze(1).long().to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item()
            
            val_loss /= len(val_loader)
            logger.info(f'Validation Loss: {val_loss:.4f}')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, f'{Config.CHECKPOINT_DIR}/voc_model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()