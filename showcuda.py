import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"可用的 CUDA 设备数量: {device_count}")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # 转换为 MB
        memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)  # 转换为 MB
        print(f"设备 {i}: {device_name}")
        print(f"已分配内存: {memory_allocated:.2f} MB")
        print(f"保留内存: {memory_reserved:.2f} MB")
else:
    print("CUDA 不可用")
