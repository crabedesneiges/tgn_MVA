import torch

# 1. Check PyTorch's compiled CUDA version (if any)
print(f"PyTorch CUDA Version: {torch.version.cuda}")

# 2. Check if the code can actually see and use an NVIDIA GPU
print(f"Is CUDA Available: {torch.cuda.is_available()}")

# 3. Check the number of usable GPUs
print(f"GPU Count: {torch.cuda.device_count()}")