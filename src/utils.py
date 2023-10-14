from torch.backends import mps

def get_device() -> str: 
    return "mps" if mps.is_available() else "cpu"