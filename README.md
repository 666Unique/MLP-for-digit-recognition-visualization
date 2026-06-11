# D2_MLP — Multi-Layer Perceptron Projects

A collection of **MLP (Multi-Layer Perceptron)** implementations for image recognition tasks, built with **PyTorch** and interactive **HTML/CSS/JavaScript** visualizations.

## 🧠 Model Architecture — `StudentMLP`

The primary model used across the number recognition tasks:

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| Flatten | Reshape | 28×28 → 784 | 0 |
| FC1 | Linear + ReLU | 784 → 128 | 100,352 |
| FC2 | Linear + ReLU | 128 → 64 | 8,256 |
| FC3 (Output) | Linear | 64 → 10 | 650 |
| **Total** | | | **109,386** |

```python
class StudentMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)        # Flatten: (batch, 1, 28, 28) → (batch, 784)
        x = self.relu(self.fc1(x))          # FC1 + ReLU: 784 → 128
        x = self.relu(self.fc2(x))          # FC2 + ReLU: 128 → 64
        x = self.fc3(x)                     # Output: 64 → 10 (logits)
        return x
```

---

## 🏋️ Training Configuration

| Parameter | Value |
|-----------|-------|
| **Dataset** | MNIST (60,000 train / 10,000 test) |
| **Optimizer** | Adam (lr = 0.001) |
| **Loss Function** | CrossEntropyLoss |
| **Batch Size** | 64 |
| **Epochs** | 5 |
| **Device** | GPU (CUDA) / CPU fallback |
| **Random Seed** | 42 |
| **Input Normalization** | mean=0.1307, std=0.3081 |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy
```

### Run Number Recognition Training

```bash
# English version (recommended)
python MLP_NumberRecognition_ENG.py
```
---

## 🌐 Interactive Visualization

The project includes a browser-based visualization tool for exploring the MLP model:

```bash
# Start a local HTTP server (required for weight loading)
python -m http.server 8000

# Then open in browser:
# http://localhost:8000/MLP_ENG_Visualization.html
```

### Visualization Features

- **🏗️ Network Structure** — Canvas-rendered diagram of the full 784→128→64→10 architecture, layer detail cards with weight shapes and parameter counts, forward propagation formula breakdown

- **📈 Training Simulation** — Animated training with start/pause/reset controls, real-time loss and accuracy charts, backpropagation visualization with gradient flow diagram, Adam optimizer update formulas

- **✏️ Prediction (Forward Pass)** — Interactive drawing canvas (mouse/touch) for digit input, 28×28 downsampling preview, actual trained weight inference with probability bars, full forward propagation trace with activation heatmaps

### Exporting Weights for HTML Visualization

The HTML visualization uses JavaScript to run inference. To export weights from a trained `.pth` file:

```bash
# Step 1: Extract weights from .pth to JSON
python extract_weights.py

# Step 2: Embed weights into HTML (makes it self-contained)
python embed_weights.py
```

> **Note:** The HTML visualization applies `transforms.Normalize(mean=0.1307, std=0.3081)` to input pixels before inference, matching the training preprocessing pipeline.

---

## 📊 Training Output

When running `MLP_NumberRecognition_ENG.py`, you'll see:

```
Current device: cuda

===== Data Sample Display =====
Batch shape: images=torch.Size([64, 1, 28, 28])
Single image shape: torch.Size([1, 28, 28]) (1 channel, 28x28 pixels)

===== Model Structure =====
Layer: fc1
Structure: Linear(in=784, out=128)
Weights shape: torch.Size([128, 784])
Bias shape: torch.Size([128])

...

Epoch 1/5 | Loss: 0.xxxxxx | Accuracy: xx.xxxx%
Epoch 2/5 | Loss: 0.xxxxxx | Accuracy: xx.xxxx%
...

Model weights saved to: mnist_mlp_weights_ENG.pth
```

---

## 📦 Saved Model Files

| File | Description |
|------|-------------|
| `mnist_mlp_weights_ENG.pth` | Trained weights for `StudentMLP` (MNIST, English version) |
| `mnist_mlp_model.pth` | Full PyTorch model (includes architecture) |
| `mnist_mlp_weights_ENG.json` | JSON export of weights (for HTML visualization) |

---

## 📝 Notes

- **Data Preprocessing:** MNIST images are normalized using dataset-specific statistics (`mean=0.1307, std=0.3081`). This normalization is critical — without it, predictions will be essentially random (~10% per class).

- **Device Selection:** The code automatically detects and uses GPU (CUDA) if available, falling back to CPU otherwise.

- **Reproducibility:** A fixed random seed (`torch.manual_seed(42)`) is set for consistent results across runs.

- **Windows Compatibility:** The `KMP_DUPLICATE_LIB_OK=TRUE` environment variable is set to resolve potential issues on Windows systems.

---

## 🛠️ Utilities

| Script | Purpose |
|--------|---------|
| `extract_weights.py` | Converts PyTorch `.pth` weight files to JSON format for use in HTML visualizations |
| `embed_weights.py` | Embeds JSON weight data directly into the HTML file for self-contained usage |

---

## 📚 Dependencies

```
torch >= 2.0
torchvision >= 0.15
matplotlib
numpy
```

---

## License

This project is for educational purposes — MLP neural network training and visualization.
