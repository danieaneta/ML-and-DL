class Config:
    # Data parameters
    DATA_DIR = 'data'
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Model parameters
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    
    # Training parameters
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    
    # Paths
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    
    # Validation
    VAL_FREQUENCY = 1  # Validate every N epochs
    
    # Logging
    SAVE_FREQUENCY = 1  # Save model every N epochs
    LOG_BATCH_FREQUENCY = 100  # Log every N batches