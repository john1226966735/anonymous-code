"""Optional check for the real CWQ/WebQSP data loader.

Requires scipy and torch-scatter, like the full training code.
"""

import os
from pathlib import Path

from load_data import DataLoader


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(Path(__file__).resolve().parent)
    loader = DataLoader("CWQ", plan_emb_dir=str(root / "embedding"))
    print("Data loader check passed.")
    print(f"train={loader.n_train}, valid={loader.n_valid}, test={loader.n_test}")
    print(f"entities={loader.n_ent}, relations={loader.n_rel}")


if __name__ == "__main__":
    main()
