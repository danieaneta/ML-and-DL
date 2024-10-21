import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from utils.data_loader import get_mnist_loaders
import logging
from tqdm import tqdm
import numpy as np

def evaluate(checkpoint_path, batch_size=32):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and load checkpoint
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get test loader
    _, test_loader = get_mnist_loaders(batch_size)

    # Evaluation loop
    total_loss = 0
    criterion = torch.nn.MSELoss()
    
    # Store some examples for visualization
    examples = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()
            
            # Store first batch of examples
            if len(examples) == 0:
                examples = [(images[:8].cpu(), outputs[:8].cpu())]

    avg_loss = total_loss / len(test_loader)
    logger.info(f'Average Test Loss: {avg_loss:.4f}')

    # Visualize results
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    
    for i in range(8):
        # Original images
        axes[0, i].imshow(examples[0][0][i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
            
        # Reconstructed images
        axes[1, i].imshow(examples[0][1][i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')

    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()

if __name__ == '__main__':
    evaluate('checkpoints/model_epoch_10.pth')