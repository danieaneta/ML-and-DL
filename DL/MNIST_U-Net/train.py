import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import UNet
from utils.data_loader import get_mnist_loaders
from tqdm import tqdm
import logging

def train(epochs=10, batch_size=32, learning_rate=0.001):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Initialize model, loss, and optimizer
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Get data loaders
    train_loader, test_loader = get_mnist_loaders(batch_size)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for images, _ in train_pbar:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)  # Autoencoder style training
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, images).item()
        
        val_loss /= len(test_loader)
        logger.info(f'Validation Loss: {val_loss:.4f}')
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, f'checkpoints/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()