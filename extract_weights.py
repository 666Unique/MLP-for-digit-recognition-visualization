import torch
import json

state = torch.load('mnist_mlp_weights_ENG.pth', map_location='cpu')

weights = {}
for k, v in state.items():
    weights[k] = v.tolist()
    print(f"{k}: shape={v.shape}, min={v.min():.6f}, max={v.max():.6f}")

with open('mnist_mlp_weights_ENG.json', 'w') as f:
    json.dump(weights, f)

print("\nWeights saved to mnist_mlp_weights_ENG.json")