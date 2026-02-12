import sys
import warnings

def check_env():
    print("Checking environment for MIAP Solver Evaluation...")

    missing = []

    # Check Torch
    try:
        import torch
        print(f"[OK] torch: {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("[FAIL] torch not found.")
        missing.append("torch")

    # Check Torch Geometric
    try:
        import torch_geometric
        try:
            print(f"[OK] torch_geometric: {torch_geometric.__version__}")
        except AttributeError:
            print(f"[OK] torch_geometric (unknown version)")
    except ImportError:
        print("[FAIL] torch_geometric not found.")
        missing.append("torch_geometric")

    # Check Ecole
    try:
        import ecole
        try:
            print(f"[OK] ecole: {ecole.__version__}")
        except AttributeError:
             print(f"[OK] ecole (unknown version)")
    except ImportError:
        print("[FAIL] ecole not found. Install via conda: `conda install -c conda-forge ecole`")
        missing.append("ecole")

    # Check PySCIPOpt
    try:
        import pyscipopt
        # PySCIPOpt version is weird sometimes
        try:
            print(f"[OK] pyscipopt: {pyscipopt.__version__}")
        except AttributeError:
             print(f"[OK] pyscipopt (unknown version)")
    except ImportError:
        print("[FAIL] pyscipopt not found. Install via conda: `conda install -c conda-forge pyscipopt`")
        missing.append("pyscipopt")

    if missing:
        print("\nERROR: Missing dependencies.")
        print(f"Please install: {', '.join(missing)}")
        sys.exit(1)
    else:
        print("\nEnvironment is ready for evaluation!")

if __name__ == "__main__":
    check_env()
