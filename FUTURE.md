# Deep Learning Framework: Future Roadmap

This document outlines the key features and architectural improvements planned for the framework.

- Normalization
    - Batch Normalization
    - Layer Normalization

- Scalers
    - Z-score scaler
    - Support excluding one-hot encoded columns from scaling. User passes one-hot indices once during fit(), and the scaler saves a persistent mask. The transform() / reverseTransform() methods then use this mask automatically for consistent behavior.

- Optimizers
    - Adam optimizer

- Execution Refactor
    - Introduce an execution object for Dense and Conv2D
    - This will handle forward and backward math cleanly, reduce large if/else chains for
    which fused kernels to call and make it easier to fuse combinations of activations, BatchNorm, and LayerNorm.

- Activations
    - SiLU
    - Sigmoid
    - GELU

- Loss Functions
    - Binary Cross-Entropy (BCE)

- Model Blocks
    - Squeeze (SE) block
    - Residual block

- Interfaces
    - Python interface for easier usage