import torch
import torch.nn as nn
import torch.nn.functional as F

class STCNNWithTransformer(nn.Module):
    def __init__(self, input_channels, num_classes, num_frames, embed_dim=128, num_heads=8, num_layers=4):
        super(STCNNWithTransformer, self).__init__()

        # Spatiotemporal Convolutions
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        
        # Transformer Encoder
        # Change the input size of the embedding layer to match the reshaped tensor
        self.embedding_layer = nn.Linear(128 * num_frames, embed_dim)  # Changed to 128 * num_frames (128 * 16 in your case) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Activation and Dropout
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.5)
        
    # The forward function should be indented to be part of the STCNNWithTransformer class
    def forward(self, x):
        # Spatiotemporal Convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        # Global Average Pooling (spatial dimensions only)
        x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))  # (Batch, Channels, Frames, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (Batch, Channels, Frames)

        # Reshape the tensor to flatten the spatial dimensions (channels * frames)
        x = x.view(x.size(0), -1)  # Flatten to (Batch, Channels * Frames)

        # Transformer Processing
        x = self.embedding_layer(x)  # Embed into Transformer input size
        x = x.unsqueeze(0)  # Add a fake sequence dimension to match (seq_len, batch, embed_dim)
        x = self.transformer_encoder(x)  # Apply Transformer
        x = x.mean(dim=0)  # Average over sequence dimension (only one frame here)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Custom Gradient Centralization Optimizer
class GCSGDM(torch.optim.SGD):
    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0):
        super().__init__(params, lr, momentum=momentum, weight_decay=weight_decay)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Gradient Centralization
                grad = p.grad.data
                if len(grad.shape) > 1:
                    grad.add_(-grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True))
                p.data.add_(-group['lr'], grad)
        return loss

# Channel Pruning (BN Scaling Factor Pruning)
def prune_channels(model, threshold=0.01):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm3d):
            # Prune channels with scaling factors below the threshold
            mask = module.weight.data.abs() > threshold
            pruned_channels = mask.sum().item()
            print(f"{name}: Pruned {module.weight.data.size(0) - pruned_channels} channels.")
            
            module.weight.data = module.weight.data[mask]
            module.bias.data = module.bias.data[mask]
            module.running_mean = module.running_mean[mask]
            module.running_var = module.running_var[mask]

# Example Usage
if __name__ == "__main__":
    # Create a model with STCNN and Transformer
    model = STCNNWithTransformer(input_channels=3, num_classes=10, num_frames=16)
    print(model)

    # Define an optimizer
    optimizer = GCSGDM(model.parameters(), lr=0.01)

    # Example input
    input_data = torch.randn(4, 3, 16, 112, 112)  # Batch size 4, 3 channels, 16 frames, 112x112 spatial resolution
    output = model(input_data)
    print("Output shape:", output.shape)

    # Prune the model
    prune_channels(model, threshold=0.01)



