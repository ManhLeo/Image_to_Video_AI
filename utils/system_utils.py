"""Runtime system inspection helpers (GPU/VRAM status)."""

import torch

def get_vram_info():
    """Returns VRAM usage info in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}
    
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    
    # Get total memory of the first GPU
    total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    
    return {
        "allocated": round(allocated, 1),
        "reserved": round(reserved, 1),
        "total": round(total, 1)
    }

def format_vram_status():
    """Returns a string status for display in UI."""
    info = get_vram_info()
    if info["total"] == 0:
        return "GPU: Not Detected / Not Available"
    return f"VRAM: {info['allocated']}MB / {info['total']}MB (Reserved: {info['reserved']}MB)"
