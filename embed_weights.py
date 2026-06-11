import json

# Read the weight data
with open('mnist_mlp_weights_ENG.json', 'r') as f:
    weights = json.load(f)

# Read the HTML file
with open('MLP_ENG_Visualization.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Build the embedded weight assignment code
weight_code = """// Load trained weights (embedded from mnist_mlp_weights_ENG.pth)
mlp.W1 = """ + json.dumps(weights['fc1.weight']) + """;
mlp.b1 = """ + json.dumps(weights['fc1.bias']) + """;
mlp.W2 = """ + json.dumps(weights['fc2.weight']) + """;
mlp.b2 = """ + json.dumps(weights['fc2.bias']) + """;
mlp.W3 = """ + json.dumps(weights['fc3.weight']) + """;
mlp.b3 = """ + json.dumps(weights['fc3.bias']) + """;
mlp.loaded = true;
document.getElementById('statusText').innerHTML = '✅ Trained weights loaded from <span class="highlight">mnist_mlp_weights_ENG.pth</span>';
console.log('Trained weights loaded successfully!');"""

# Replace the fetch-based loading with embedded weights
old_code = """// Load trained weights from mnist_mlp_weights_ENG.pth (exported as JSON)
mlp.loadFromJSON('mnist_mlp_weights_ENG.json');"""

html = html.replace(old_code, weight_code)

# Write the updated HTML
with open('MLP_ENG_Visualization.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Weights embedded successfully!")
print("HTML file size:", len(html), "bytes")