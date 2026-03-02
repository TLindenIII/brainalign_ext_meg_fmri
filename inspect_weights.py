import torch
weights = torch.load('src/vendored/CBraMod/pretrained_weights/pretrained_weights.pth', map_location='cpu')
for k, v in weights.items():
    if len(v.shape) > 0:
        print(f"{k}: {v.shape}")
    else:
        print(f"{k}: scalar")
