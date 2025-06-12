from .prediction import predict_one, predict_volume

try:
    import torch
except ImportError:
    raise ImportError("Please install PyTorch to use the vseg segmentation module of the LIOM toolkit.")
