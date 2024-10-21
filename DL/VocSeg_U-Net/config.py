class Config:
    # Data parameters
    DATA_DIR = 'data'
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    IMAGE_SIZE = 256

    # Model parameters
    IN_CHANNELS = 3
    OUT_CHANNELS = 21
    NUM_CLASSES = 21
    DEVICE = 'cuda'

    # Training parameters
    EPOCHS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # Paths
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    
    # Validation
    VAL_FREQUENCY = 1