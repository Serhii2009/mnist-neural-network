import os
import glob
import struct
from typing import Tuple, List
import numpy as np

# ---------------- KaggleHub ----------------
def try_kagglehub_download() -> str:
    try:
        import kagglehub
        path = kagglehub.dataset_download("hojjatk/mnist-dataset")
        print("Path to dataset files:", path)
        return path
    except Exception as e:
        print("kagglehub not available or download failed:", e)
        return ""

# ---------------- Basic helpers ----------------
def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    y = np.zeros((labels.size, num_classes), dtype=np.float32)
    y[np.arange(labels.size), labels.astype(int)] = 1.0
    return y

def normalize_pixels(x: np.ndarray) -> np.ndarray:
    return (x / 255.0).astype(np.float32)

def to_column_vectors(X: np.ndarray) -> List[np.ndarray]:
    return [X[i].reshape(-1, 1) for i in range(X.shape[0])]

def to_training_list(X: np.ndarray, y_onehot: np.ndarray):
    X_cols = to_column_vectors(X)
    Y_cols = [y_onehot[i].reshape(-1, 1) for i in range(y_onehot.shape[0])]
    return list(zip(X_cols, Y_cols))

def to_eval_list(X: np.ndarray, y: np.ndarray):
    X_cols = to_column_vectors(X)
    if y is None:
        return [(x, None) for x in X_cols]
    if y.ndim == 2 and y.shape[1] == 10:
        Y_cols = [y[i].reshape(-1, 1) for i in range(y.shape[0])]
    else:
        Y_cols = [np.array([[int(y[i])]], dtype=np.int32) for i in range(y.shape[0])]
    return list(zip(X_cols, Y_cols))

# ---------------- CSV loader ----------------
def load_from_csv(csv_path: str, has_labels: bool = True):
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]
    if has_labels:
        labels = data[:, 0].astype(int)
        pixels = data[:, 1:]
        return pixels, labels
    else:
        pixels = data
        return pixels, None

def discover_csvs(base_dir: str):
    patterns = [
        os.path.join(base_dir, "*train*.csv"),
        os.path.join(base_dir, "*test*.csv"),
    ]
    found = []
    for p in patterns:
        found.extend(glob.glob(p))
    return found

# ---------------- NPY loader ----------------
def try_load_npy(data_dir: str):
    paths = {
        "x_train": os.path.join(data_dir, "x_train.npy"),
        "y_train": os.path.join(data_dir, "y_train.npy"),
        "x_test":  os.path.join(data_dir, "x_test.npy"),
        "y_test":  os.path.join(data_dir, "y_test.npy"),
    }
    if all(os.path.exists(p) for p in paths.values()):
        x_train = np.load(paths["x_train"])
        y_train = np.load(paths["y_train"])
        x_test  = np.load(paths["x_test"])
        y_test  = np.load(paths["y_test"])
        return x_train, y_train, x_test, y_test
    return None

# ---------------- IDX loader ----------------
def load_idx_images(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols)

def load_idx_labels(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic {magic} in {path}")
        return np.frombuffer(f.read(), dtype=np.uint8)

def try_load_idx(data_dir: str):
    paths = {
        "train_images": os.path.join(data_dir, "train-images.idx3-ubyte"),
        "train_labels": os.path.join(data_dir, "train-labels.idx1-ubyte"),
        "test_images":  os.path.join(data_dir, "t10k-images.idx3-ubyte"),
        "test_labels":  os.path.join(data_dir, "t10k-labels.idx1-ubyte"),
    }
    if all(os.path.exists(p) for p in paths.values()):
        Xtr = load_idx_images(paths["train_images"])
        ytr = load_idx_labels(paths["train_labels"])
        Xte = load_idx_images(paths["test_images"])
        yte = load_idx_labels(paths["test_labels"])
        return Xtr, ytr, Xte, yte
    return None

# ---------------- Main loader ----------------
def load_mnist(data_dir: str = "data") -> Tuple[List, List]:
    """
    Returns:
      training_data: list[(x:(784,1), y:(10,1))]
      test_data:     list[(x:(784,1), y:int or one-hot)]
    Load order preference:
      1) KaggleHub CSVs
      2) Local CSVs
      3) Local NPYS
      4) IDX (classic MNIST format)
    """
    # 1) KaggleHub
    kaggle_path = try_kagglehub_download()
    csv_candidates = []
    if kaggle_path:
        csv_candidates.extend(discover_csvs(kaggle_path))

    # 2) Local CSVs
    csv_candidates.extend(discover_csvs(data_dir))

    x_train = y_train = x_test = y_test = None

    # --- Try CSV ---
    train_csv = None
    test_csv = None
    for p in csv_candidates:
        name = os.path.basename(p).lower()
        if "train" in name and train_csv is None:
            train_csv = p
        elif "test" in name and test_csv is None:
            test_csv = p

    if train_csv:
        Xtr, ytr = load_from_csv(train_csv, has_labels=True)
        Xtr = normalize_pixels(Xtr)
        ytr_oh = one_hot(ytr, 10)
        training_data = to_training_list(Xtr, ytr_oh)
    else:
        # --- Try NPY ---
        npy_pack = try_load_npy(data_dir)
        if npy_pack is not None:
            x_train, y_train, x_test, y_test = npy_pack
            Xtr = normalize_pixels(x_train.reshape(-1, 784))
            if y_train.ndim == 2 and y_train.shape[1] == 10:
                ytr_oh = y_train.astype(np.float32)
            else:
                ytr_oh = one_hot(y_train, 10)
            training_data = to_training_list(Xtr, ytr_oh)
        else:
            # --- Try IDX ---
            idx_pack = try_load_idx(data_dir)
            if idx_pack is not None:
                x_train, y_train, x_test, y_test = idx_pack
                Xtr = normalize_pixels(x_train)
                ytr_oh = one_hot(y_train, 10)
                training_data = to_training_list(Xtr, ytr_oh)
            else:
                raise FileNotFoundError(
                    "Could not find MNIST data. Provide Kaggle CSVs, local CSVs, "
                    "NPY files, or IDX files in ./data."
                )

    # --- Test data ---
    if test_csv:
        Xte, yte = load_from_csv(test_csv, has_labels=True)
        if yte is None or (yte.size == 0):
            Xte, yte = load_from_csv(test_csv, has_labels=False)
        Xte = normalize_pixels(Xte)
        test_data = to_eval_list(Xte, yte if yte is not None else None)
    elif x_test is not None:
        Xte = normalize_pixels(x_test.reshape(-1, 784))
        test_data = to_eval_list(Xte, y_test)
    else:
        test_data = []

    return training_data, test_data
