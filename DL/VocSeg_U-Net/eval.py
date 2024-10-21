import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from utils.data_loader import get_voc_loaders
import logging
from tqdm import tqdm
import numpy as np
from config import Config

def evaluate(checkpoint_path):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device(Config.DEVICE)
    model = UNet(in_channels=Config.IN_CHANNELS, out_channels=Config.NUM_CLASSES).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _, val_loader = get_voc_loaders(Config.DATA_DIR, batch_size=4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluation metrics
    total_loss = 0
    total_iou = 0
    num_batches = len(val_loader)
    
    # Store examples for visualization
    sample_images = []
    sample_masks = []
    sample_predictions = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(val_loader, desc='Evaluating')):
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            print(f"Output shape: {outputs.shape}")
            print(f"Output range: {outputs.min().item()} to {outputs.max().item()}")
            
            # Calculate IoU
            predictions = torch.argmax(outputs, dim=1)
            intersection = torch.logical_and(predictions, masks)
            union = torch.logical_or(predictions, masks)
            batch_iou = torch.sum(intersection) / torch.sum(union)
            total_iou += batch_iou.item()
            
            # Store first batch for visualization
            if i == 0:
                sample_images = images.cpu()[:4]
                sample_masks = masks.cpu()[:4]
                sample_predictions = predictions.cpu()[:4]

    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    logger.info(f'Average Loss: {avg_loss:.4f}')
    logger.info(f'Average IoU: {avg_iou:.4f}')

    # Visualize results
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for idx in range(4):
        # Original image
        axes[0, idx].imshow(sample_images[idx].permute(1, 2, 0))
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_title('Input Image')
        
        # Ground truth
        axes[1, idx].imshow(sample_masks[idx])
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].set_title('Ground Truth')
        
        # Prediction
        axes[2, idx].imshow(sample_predictions[idx])
        axes[2, idx].axis('off')
        if idx == 0:
            axes[2, idx].set_title('Prediction')

    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()

if __name__ == '__main__':
    evaluate('checkpoints/voc_model_epoch_10.pth')  # Or whatever checkpoint you want to evaluate