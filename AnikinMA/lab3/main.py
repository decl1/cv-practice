import os
import argparse
from dataset import build_splits
from pipeline import BoWClassifier, NNClassifier, print_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split_train", type=str, required=True)
    ap.add_argument("--split_test", type=str, default=None)
    ap.add_argument("--mode", type=str, default="train_test", choices=["train", "test", "train_test"])
    ap.add_argument("--algo", type=str, default="bow", choices=["bow", "nn"])
    ap.add_argument("--artifacts", type=str, default="artifacts")

    ap.add_argument("--detector", type=str, default="sift")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--svm_kernel", type=str, default="linear", choices=["linear", "rbf"])
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--max_side", type=int, default=640)
    ap.add_argument("--nfeatures", type=int, default=2000)
    ap.add_argument("--l2_norm", type=int, default=1)

    ap.add_argument("--backbone", type=str, default="resnet18")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--freeze", type=int, default=1)

    args = ap.parse_args()

    Xtr, ytr, Xte, yte, classes = build_splits(args.data_dir, args.split_train, args.split_test)

    os.makedirs(args.artifacts, exist_ok=True)

    if args.algo == "bow":
        model = BoWClassifier(
            detector=args.detector,
            k=args.k,
            svm_kernel=args.svm_kernel,
            C=args.C,
            gamma=args.gamma,
            max_side=args.max_side,
            nfeatures=args.nfeatures,
            l2_norm=args.l2_norm
        )
        if args.mode in ("train", "train_test"):
            model.fit(Xtr, ytr)
            model.save(args.artifacts)
        if args.mode in ("test", "train_test"):
            if args.mode == "test":
                model.load(args.artifacts)
            preds = model.predict(Xte)
            print_report(yte, preds, classes)

    else:
        model = NNClassifier(
            classes=classes,
            backbone=args.backbone,
            img_size=args.img_size,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            freeze=args.freeze
        )
        if args.mode in ("train", "train_test"):
            model.fit(Xtr, ytr, X_val=Xte, y_val=yte)
            model.save(args.artifacts)
        if args.mode in ("test", "train_test"):
            if args.mode == "test":
                model.load(args.artifacts)
            preds = model.predict(Xte)
            print_report(yte, preds, classes)

if __name__ == "__main__":
    main()