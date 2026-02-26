import os
import glob

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

DEFAULT_CLASSES = ["01_nizhnynovgorodkremlin", "08_palaceoflabor", "04_arkhangelskcathedral"]

DEFAULT_PATTERNS = {
    "01_nizhnynovgorodkremlin": ["01_nizhnynovgorodkremlin"],
    "08_palaceoflabor": ["08_palaceoflabor"],
    "04_arkhangelskcathedral": ["04_arkhangelskcathedral"]
}

def _norm_rel(p: str) -> str:
    return p.replace("\\", "/").strip()

def read_split_file(path: str):
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(_norm_rel(s))
    return out

def infer_label_from_path(path: str, class_patterns=None):
    p = _norm_rel(path).lower()
    class_patterns = class_patterns or DEFAULT_PATTERNS
    for cls, pats in class_patterns.items():
        for pat in pats:
            if pat.lower() in p:
                return cls
    return None

def list_all_images(data_dir: str):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(data_dir, "**", f"*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(data_dir, "**", f"*{ext.upper()}"), recursive=True))
    files = sorted(list(set(files)))
    return files

def build_splits(data_dir: str, split_train: str, split_test: str = None, class_patterns=None):
    all_imgs = list_all_images(data_dir)
    if len(all_imgs) == 0:
        raise RuntimeError("Не найдено изображений в data_dir")

    train_list = read_split_file(split_train)
    if train_list is None or len(train_list) == 0:
        raise RuntimeError("Файл train-разбиения пустой или не задан")

    train_set = set(train_list)

    def resolve_path(rel_or_abs: str):
        rel_or_abs_n = _norm_rel(rel_or_abs)
        if os.path.isabs(rel_or_abs_n) and os.path.isfile(rel_or_abs_n):
            return rel_or_abs_n
        cand = os.path.join(data_dir, rel_or_abs_n)
        if os.path.isfile(cand):
            return cand
        base = os.path.basename(rel_or_abs_n)
        hits = [p for p in all_imgs if os.path.basename(p) == base]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return hits[0]
        return None

    train_paths = []
    for item in train_list:
        rp = resolve_path(item)
        if rp is not None:
            train_paths.append(rp)

    if split_test is not None:
        test_list = read_split_file(split_test) or []
        test_paths = []
        for item in test_list:
            rp = resolve_path(item)
            if rp is not None:
                test_paths.append(rp)
    else:
        train_resolved_rel = set(_norm_rel(os.path.relpath(p, data_dir)) for p in train_paths)
        test_paths = []
        for p in all_imgs:
            rel = _norm_rel(os.path.relpath(p, data_dir))
            if rel not in train_set and rel not in train_resolved_rel:
                test_paths.append(p)

    class_patterns = class_patterns or DEFAULT_PATTERNS
    classes = list(DEFAULT_CLASSES)

    def label_to_id(lbl: str):
        return classes.index(lbl)

    def make_xy(paths):
        X, y = [], []
        for p in paths:
            lbl = infer_label_from_path(p, class_patterns=class_patterns)
            if lbl is None:
                continue
            if lbl not in classes:
                continue
            X.append(p)
            y.append(label_to_id(lbl))
        return X, y

    Xtr, ytr = make_xy(train_paths)
    Xte, yte = make_xy(test_paths)

    if len(Xtr) == 0:
        raise RuntimeError("Не удалось собрать train-выборку: проверь разбиение и структуру папок/названия")
    if len(Xte) == 0:
        raise RuntimeError("Не удалось собрать test-выборку: проверь разбиение и структуру папок/названия")

    return Xtr, ytr, Xte, yte, classes