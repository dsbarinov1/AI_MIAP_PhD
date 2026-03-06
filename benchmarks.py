"""
Loaders for MIAP/3IAP benchmark instances from the literature.
- Crama-Spieksma (EJOR 1992): 18 instances, n=33 or n=66, three matrices c_IJ, c_IK, c_JK.
  Format: first line "3", second line n, then three n×n matrices separated by blank lines.
  Cost is decomposable: c_ijk = c_IJ[i,j] + c_IK[i,k] + c_JK[j,k].
- Balas-Saltzman: optional; same 3-index formulation (loader placeholder for future use).
"""
import os
import numpy as np


def _read_matrix_lines(lines, start_idx, n):
    """Read n*n floats from lines starting at start_idx; return (matrix 2D, next_index)."""
    numbers = []
    idx = start_idx
    while len(numbers) < n * n and idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line:
            continue
        for tok in line.split():
            try:
                numbers.append(float(tok))
            except ValueError:
                continue
    if len(numbers) < n * n:
        raise ValueError(f"Expected {n * n} numbers for matrix, got {len(numbers)}")
    mat = np.array(numbers[: n * n], dtype=np.float64).reshape(n, n)
    return mat, idx


def load_crama_spieksma(filepath: str, scale: float = 1.0):
    """
    Load one Crama-Spieksma instance (EJOR 1992 format).
    File format: line 1 = "3", line 2 = n, then three n×n matrices (c_IJ, c_IK, c_JK) separated by blank lines.
    Cost tensor: cost[i,j,k] = c_IJ[i,j] + c_IK[i,k] + c_JK[j,k].
    scale: multiply all costs by this (e.g. 0.01 if file values are Table 1 × 100).
    Returns:
        n: int
        cost_tensor: np.ndarray shape (n, n, n)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # first line: 3
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise ValueError("File empty or only blanks")
    k = int(lines[idx].strip())
    idx += 1
    if k != 3:
        raise ValueError("Expected first line 3 for 3IAP")
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise ValueError("Missing n")
    n = int(lines[idx].strip())
    idx += 1

    c_IJ, idx = _read_matrix_lines(lines, idx, n)
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    c_IK, idx = _read_matrix_lines(lines, idx, n)
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    c_JK, idx = _read_matrix_lines(lines, idx, n)

    c_IJ = c_IJ * scale
    c_IK = c_IK * scale
    c_JK = c_JK * scale

    cost_tensor = np.zeros((n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                cost_tensor[i, j, k] = c_IJ[i, j] + c_IK[i, k] + c_JK[j, k]

    return n, cost_tensor


def load_crama_spieksma_directory(dirpath: str, scale: float = 1.0):
    """
    Load all Crama-Spieksma .txt instances from a directory.
    Returns list of (name, n, cost_tensor).
    """
    out = []
    for fname in sorted(os.listdir(dirpath)):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(dirpath, fname)
        if not os.path.isfile(path):
            continue
        try:
            n, cost_tensor = load_crama_spieksma(path, scale=scale)
            out.append((fname.replace(".txt", ""), n, cost_tensor))
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}") from e
    return out


def load_balas_saltzman(filepath: str):
    """
    Placeholder for Balas-Saltzman format loader.
    When implemented: parse file and return (n, cost_tensor) for 3IAP.
    """
    raise NotImplementedError(
        "Balas-Saltzman loader not implemented; use load_crama_spieksma or add format from Three-IAP repo."
    )
