import sys
import torch

print("="*40)
print(f"Python: {sys.version.split()[0]}")
print(f"Torch:  {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version (Torch): {torch.version.cuda}")
else:
    print("!!! WARNING: CPU ONLY MODE !!!")

try:
    import ecole
    import pyscipopt
    print("Solver: OK (Ecole + PySCIPOpt found)")
except ImportError as e:
    print(f"Solver: FAIL ({e})")

try:
    import torch_geometric
    print("PyG:    OK")
except ImportError:
    print("PyG:    FAIL")
print("="*40)