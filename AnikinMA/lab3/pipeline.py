import os
import time
import json
import numpy as np
import cv2

def set_seed(seed: int):
    np.random.seed(seed)

def load_image_bgr(path: str, max_side: int = None):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if max_side is not None:
        h, w = img.shape[:2]
        s = max(h, w)
        if s > max_side:
            scale = max_side / float(s)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def confusion_matrix(y_true, y_pred, n_classes):
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m

def print_report(y_true, y_pred, classes):
    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, len(classes))
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    return acc, cm

def _create_feature(detector_name: str, nfeatures: int = 2000):
    name = (detector_name or "").lower()
    if name == "sift":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=nfeatures), "float"
        if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
            return cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures), "float"
        name = "orb"
    if name == "akaze":
        return cv2.AKAZE_create(), "binary"
    return cv2.ORB_create(nfeatures=nfeatures), "binary"

def _stack_desc(desc_list, max_total: int = 200000):
    all_desc = [d for d in desc_list if d is not None and len(d) > 0]
    if len(all_desc) == 0:
        return None
    D = np.vstack(all_desc)
    if D.shape[0] > max_total:
        idx = np.random.choice(D.shape[0], size=max_total, replace=False)
        D = D[idx]
    return D

def _kmeans_vocab(descs_float32, k: int, attempts: int = 3, max_iter: int = 50, eps: float = 1e-3):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(descs_float32, k, None, criteria, attempts, flags)
    return centers.astype(np.float32)

def _assign_to_vocab(descs, centers, dtype_kind: str):
    if descs is None or len(descs) == 0:
        return None
    if dtype_kind == "binary":
        descs = descs.astype(np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    centers_d = centers.astype(np.float32)
    matches = matcher.match(descs.astype(np.float32), centers_d)
    idx = np.array([m.trainIdx for m in matches], dtype=np.int32)
    return idx

def _hist_from_assign(assign_idx, k: int):
    h = np.zeros((k,), dtype=np.float32)
    if assign_idx is None or len(assign_idx) == 0:
        return h
    vals, cnts = np.unique(assign_idx, return_counts=True)
    h[vals] = cnts.astype(np.float32)
    return h

def _l2norm(x, eps=1e-12):
    n = np.sqrt((x * x).sum()) + eps
    return x / n

class BoWClassifier:
    def __init__(self, detector="sift", k=256, svm_kernel="linear", C=2.0, gamma=0.5, max_side=640, nfeatures=2000, l2_norm=1, seed=42):
        self.detector_name = detector
        self.k = int(k)
        self.svm_kernel = (svm_kernel or "linear").lower()
        self.C = float(C)
        self.gamma = float(gamma)
        self.max_side = int(max_side) if max_side is not None else None
        self.nfeatures = int(nfeatures)
        self.l2_norm = int(l2_norm)
        self.seed = int(seed)
        self.detector, self.dtype_kind = _create_feature(self.detector_name, nfeatures=self.nfeatures)
        self.centers = None
        self.svm = None

    def _extract_desc(self, path):
        img = load_image_bgr(path, max_side=self.max_side)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = self.detector.detectAndCompute(gray, None)
        return desc

    def fit(self, X_paths, y):
        set_seed(self.seed)
        t0 = time.time()
        descs = []
        for p in X_paths:
            d = self._extract_desc(p)
            if d is not None and len(d) > 0:
                if self.dtype_kind == "binary":
                    d = d.astype(np.uint8)
                descs.append(d)
        D = _stack_desc(descs)
        if D is None or D.shape[0] < max(10, self.k):
            raise RuntimeError("Недостаточно дескрипторов для построения словаря")
        if self.dtype_kind == "binary":
            D = D.astype(np.float32)
        else:
            D = D.astype(np.float32)
        self.centers = _kmeans_vocab(D, self.k)

        H = []
        for p in X_paths:
            d = self._extract_desc(p)
            if d is None or len(d) == 0:
                h = np.zeros((self.k,), dtype=np.float32)
            else:
                a = _assign_to_vocab(d, self.centers, self.dtype_kind)
                h = _hist_from_assign(a, self.k)
            if self.l2_norm:
                h = _l2norm(h)
            H.append(h)
        H = np.vstack(H).astype(np.float32)
        y = np.asarray(y, dtype=np.int32).reshape(-1, 1)

        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        if self.svm_kernel == "rbf":
            svm.setKernel(cv2.ml.SVM_RBF)
            svm.setGamma(self.gamma)
        else:
            svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setC(self.C)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 2000, 1e-6))
        svm.train(H, cv2.ml.ROW_SAMPLE, y)
        self.svm = svm
        t1 = time.time()
        print(f"BoW train done in {t1 - t0:.2f}s, vocab={self.k}, detector={self.detector_name}, kernel={self.svm_kernel}")

    def predict(self, X_paths):
        if self.centers is None or self.svm is None:
            raise RuntimeError("Модель не обучена")
        H = []
        for p in X_paths:
            d = self._extract_desc(p)
            if d is None or len(d) == 0:
                h = np.zeros((self.k,), dtype=np.float32)
            else:
                a = _assign_to_vocab(d, self.centers, self.dtype_kind)
                h = _hist_from_assign(a, self.k)
            if self.l2_norm:
                h = _l2norm(h)
            H.append(h)
        H = np.vstack(H).astype(np.float32)
        _, resp = self.svm.predict(H)
        return resp.reshape(-1).astype(np.int32).tolist()

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "bow_centers.npy"), self.centers)
        self.svm.save(os.path.join(out_dir, "bow_svm.xml"))
        meta = {
            "detector": self.detector_name,
            "k": self.k,
            "svm_kernel": self.svm_kernel,
            "C": self.C,
            "gamma": self.gamma,
            "max_side": self.max_side,
            "nfeatures": self.nfeatures,
            "l2_norm": self.l2_norm,
            "seed": self.seed,
            "dtype_kind": self.dtype_kind
        }
        with open(os.path.join(out_dir, "bow_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, out_dir: str):
        centers_path = os.path.join(out_dir, "bow_centers.npy")
        svm_path = os.path.join(out_dir, "bow_svm.xml")
        meta_path = os.path.join(out_dir, "bow_meta.json")
        self.centers = np.load(centers_path).astype(np.float32)
        self.svm = cv2.ml.SVM_load(svm_path)
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.detector_name = meta.get("detector", self.detector_name)
            self.k = int(meta.get("k", self.k))
            self.svm_kernel = meta.get("svm_kernel", self.svm_kernel)
            self.C = float(meta.get("C", self.C))
            self.gamma = float(meta.get("gamma", self.gamma))
            self.max_side = meta.get("max_side", self.max_side)
            self.nfeatures = int(meta.get("nfeatures", self.nfeatures))
            self.l2_norm = int(meta.get("l2_norm", self.l2_norm))
            self.seed = int(meta.get("seed", self.seed))
            self.detector, self.dtype_kind = _create_feature(self.detector_name, nfeatures=self.nfeatures)

class NNClassifier:
    def __init__(self, classes, backbone="resnet18", img_size=224, epochs=10, lr=1e-3, batch_size=16, freeze=1, seed=42, device=None):
        self.classes = list(classes)
        self.backbone = backbone
        self.img_size = int(img_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.freeze = int(freeze)
        self.seed = int(seed)
        self.device = device
        self.model = None

    def _imports(self):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        import torchvision
        import torchvision.transforms as T
        return torch, nn, optim, Dataset, DataLoader, torchvision, T

    def _build_model(self, torchvision, nn):
        name = (self.backbone or "").lower()
        if name == "mobilenet_v3_small":
            m = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
            in_f = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_f, len(self.classes))
            return m
        if name == "mobilenet_v3_large":
            m = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
            in_f = m.classifier[-1].in_features
            m.classifier[-1] = nn.Linear(in_f, len(self.classes))
            return m
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, len(self.classes))
        return m

    def fit(self, X_paths, y, X_val=None, y_val=None):
        torch, nn, optim, Dataset, DataLoader, torchvision, T = self._imports()
        set_seed(self.seed)
        torch.manual_seed(self.seed)

        class CvDataset(Dataset):
            def __init__(self, paths, labels, img_size, train):
                self.paths = list(paths)
                self.labels = list(labels)
                self.img_size = img_size
                self.train = train
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                p = self.paths[idx]
                img = load_image_bgr(p, max_side=None)
                if img is None:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                if self.train and np.random.rand() < 0.5:
                    img = img[:, ::-1, :]
                x = img.astype(np.float32) / 255.0
                x = (x - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
                x = np.transpose(x, (2, 0, 1))
                return torch.from_numpy(x), torch.tensor(int(self.labels[idx]), dtype=torch.long)

        ds = CvDataset(X_paths, y, self.img_size, train=True)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        if X_val is not None and y_val is not None:
            ds_val = CvDataset(X_val, y_val, self.img_size, train=False)
            dl_val = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False, num_workers=0)
        else:
            dl_val = None

        model = self._build_model(torchvision, nn)

        if self.freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                if name.endswith("fc.weight") or name.endswith("fc.bias") or "classifier" in name:
                    param.requires_grad = True

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model = model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        opt = optim.Adam(params, lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, self.epochs + 1):
            model.train()
            total_loss = 0.0
            n = 0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)
            tr_loss = total_loss / max(1, n)

            if dl_val is not None:
                model.eval()
                preds = []
                gts = []
                with torch.no_grad():
                    for xb, yb in dl_val:
                        xb = xb.to(device)
                        out = model(xb)
                        pr = torch.argmax(out, dim=1).cpu().numpy().tolist()
                        preds.extend(pr)
                        gts.extend(yb.numpy().tolist())
                acc = accuracy(gts, preds)
                print(f"Epoch {epoch}/{self.epochs} loss={tr_loss:.4f} val_acc={acc:.4f}")
            else:
                print(f"Epoch {epoch}/{self.epochs} loss={tr_loss:.4f}")

        self.model = model

    def predict(self, X_paths):
        torch, nn, optim, Dataset, DataLoader, torchvision, T = self._imports()
        if self.model is None:
            raise RuntimeError("Модель не обучена")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        class CvDataset(Dataset):
            def __init__(self, paths, img_size):
                self.paths = list(paths)
                self.img_size = img_size
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                p = self.paths[idx]
                img = load_image_bgr(p, max_side=None)
                if img is None:
                    img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                x = img.astype(np.float32) / 255.0
                x = (x - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
                x = np.transpose(x, (2, 0, 1))
                return torch.from_numpy(x)

        ds = CvDataset(X_paths, self.img_size)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb in dl:
                xb = xb.to(device)
                out = self.model(xb)
                pr = torch.argmax(out, dim=1).cpu().numpy().tolist()
                preds.extend(pr)
        return preds

    def save(self, out_dir: str):
        import torch
        os.makedirs(out_dir, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "classes": self.classes,
            "backbone": self.backbone,
            "img_size": self.img_size
        }, os.path.join(out_dir, "nn.pt"))

    def load(self, out_dir: str):
        import torch
        import torch.nn as nn
        import torchvision
        ckpt = torch.load(os.path.join(out_dir, "nn.pt"), map_location="cpu")
        self.classes = ckpt["classes"]
        self.backbone = ckpt.get("backbone", self.backbone)
        self.img_size = int(ckpt.get("img_size", self.img_size))
        self.model = self._build_model(torchvision, nn)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()