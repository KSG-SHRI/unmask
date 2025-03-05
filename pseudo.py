import torch
print(torch.cuda.is_available()) 
print(torch.cuda.device_count())
print("CUDA Version:", torch.version.cuda) 
# if the cuda is False uninstall reinstall -- for cuda 12.1

# pip uninstall torch torchvision torchaudio
# pip cache purge
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
