# MLP-for-digit-recognition
- __🏗️ Network Structure__ — Canvas-rendered diagram of `StudentMLP` (784→128→64→10), layer detail cards with weight shapes and PyTorch code references, model configuration panel (Adam lr=0.001, CrossEntropyLoss, batch_size=64, MNIST normalization), forward propagation formula breakdown

- __📈 Training Simulation__ — Animated training with start/pause/reset, configurable epochs (default 5), real-time loss & accuracy charts, backpropagation visualization with forward/backward pass diagram, Adam optimizer update formulas, epoch-by-epoch training log

- __✏️ Prediction (Forward Pass)__ — Drawing canvas with 28×28 downsampling, __trained weights loaded from `mnist_mlp_weights_ENG.pth`__, probability bar chart for digits 0-9, full forward propagation trace with activation heatmaps at each layer, random digit generator
