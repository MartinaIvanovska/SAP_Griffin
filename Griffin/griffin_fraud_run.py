#!/usr/bin/env python

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, copy, json, os.path as osp, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from hdataset import Graph, Task
from hloaderwrapper import LoaderWrapperTask, buildindice
from hmodel import GriffinMod
from hFloatEmb import SimpleRepeater


def _first(xlist):
    return xlist[0]


def to_device(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(i, dev) for i in x)
    if isinstance(x, dict):
        return {k: to_device(v, dev) for k, v in x.items()}
    return x


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_checkpoint(path):
    if osp.isfile(path):
        return path
    for cand in (osp.join(path, "best_checkpoint", "model.safetensors"), osp.join(path, "model.safetensors")):
        if osp.isfile(cand):
            return cand
    hits = sorted(os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs if f == "model.safetensors")
    if not hits:
        raise SystemExit(f"no model.safetensors found under {path}")
    hits.sort(key=lambda h: ("best_checkpoint" not in h, len(h)))
    return hits[0]


def load_pretrained(model, path):
    f = find_checkpoint(path)
    from safetensors.torch import load_file
    sd = load_file(f)
    res = model.load_state_dict(sd, strict=False)
    print(f"loaded {f}: {len(sd)} tensors, missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
    if res.missing_keys or res.unexpected_keys:
        print("  missing   :", res.missing_keys[:6])
        print("  unexpected:", res.unexpected_keys[:6])
    if len(res.missing_keys) > 0.2 * len(model.state_dict()):
        raise SystemExit("Checkpoint does not match the architecture (try --use_gate False/True, --hiddim, --num_mp).")


def make_ds(graph, task, split, args, floatemb):
    return LoaderWrapperTask(
        graph, batch_size=args.batchsize,
        subgraphargs={"floatemb": floatemb, "fanout": args.fanout, "hop": args.hop},
        shuffle=(split == "train"), task=task, tasknames=[args.taskname], split=split,
        fewshotfanout=args.fewshotfanout)


def forward(model, data):
    label, y, mapping = data[-3:]
    out = model(*data[:-3])[mapping] @ y.T
    return out, label


@torch.no_grad()
def predict(model, ds, dev, repeats, seed):
    model.eval()
    ds.ind = buildindice(False, ds.lens, ds.batch_size)
    probs, labels = [], None
    for r in range(repeats):
        torch.manual_seed(seed + r); np.random.seed(seed + r)
        outs, labs = [], []
        for data in DataLoader(ds, shuffle=False, batch_size=1, collate_fn=_first, num_workers=0):
            out, lab = forward(model, to_device(data, dev))
            outs.append(out.float().cpu()); labs.append(lab.cpu())
        probs.append(torch.softmax(torch.cat(outs), dim=-1))
        labels = torch.cat(labs).numpy()
    return torch.stack(probs).mean(0).numpy(), labels


def score(probs, labels):
    p1 = probs[:, 1]
    return roc_auc_score(labels, p1), average_precision_score(labels, p1)


def best_f1_threshold(p1, y):
    best_t, best_f = 0.5, -1.0
    for t in np.unique(np.round(p1, 6)):
        pred = p1 >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        f = 2 * tp / max(2 * tp + fp + fn, 1)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="output dir of build_griffin_dataset.py")
    ap.add_argument("--loadpath", default=None, help="folder of the pretrained single-sft checkpoint (contains model.safetensors)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke test / ablation)")
    ap.add_argument("--outdir", default="runs/fraud")
    ap.add_argument("--taskname", default="creditcard-Transactions-Class")
    ap.add_argument("--epochs", type=int, default=30, help="0 = evaluate the pretrained model without fine-tuning")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batchsize", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=2e-4)
    ap.add_argument("--fraud-weight", type=float, default=1.0, help="loss weight of class 1 (1.0 = repo default)")
    ap.add_argument("--hop", type=int, default=0)
    ap.add_argument("--fanout", type=int, default=20)
    ap.add_argument("--fewshotfanout", type=int, default=3)
    ap.add_argument("--hiddim", type=int, default=512)
    ap.add_argument("--num_mp", type=int, default=4)
    ap.add_argument("--use_rev", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    ap.add_argument("--use_gate", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    ap.add_argument("--eval-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = pick_device(args.device)
    print("device:", dev)

    model = GriffinMod(hiddim=args.hiddim, num_mp=args.num_mp, use_rev=args.use_rev, use_gate=args.use_gate)
    if not args.no_pretrained:
        if not args.loadpath:
            raise SystemExit("give --loadpath <checkpoints/single-sft> or --no-pretrained")
        load_pretrained(model, args.loadpath)
    model.to(dev)

    graph, task = Graph(args.dataset), Task(args.dataset)
    floatemb = SimpleRepeater(args.hiddim)
    ds_tr, ds_va, ds_te = (make_ds(graph, task, s, args, floatemb) for s in ("train", "valid", "test"))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    cw = torch.tensor([1.0, args.fraud_weight], device=dev)

    va_probs, va_y = predict(model, ds_va, dev, args.eval_repeats, args.seed)
    auc, ap_ = score(va_probs, va_y)
    best = {"auc": auc, "ap": ap_, "epoch": -1, "state": copy.deepcopy(model.state_dict())}
    print(f"[before fine-tuning] val AUROC {auc:.4f}  AP {ap_:.4f}")
    history = [{"epoch": -1, "val_auroc": auc, "val_ap": ap_}]

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ds_tr.ind = buildindice(True, ds_tr.lens, ds_tr.batch_size)
        losses = []
        for data in DataLoader(ds_tr, shuffle=True, batch_size=1, collate_fn=_first, num_workers=0):
            out, lab = forward(model, to_device(data, dev))
            loss = F.cross_entropy(out, lab, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        va_probs, va_y = predict(model, ds_va, dev, 1, args.seed + epoch)
        auc, ap_ = score(va_probs, va_y)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auroc": auc, "val_ap": ap_})
        flag = ""
        if (auc, ap_) > (best["auc"], best["ap"]):
            best = {"auc": auc, "ap": ap_, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            flag = "  <- best"
        print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  val AUROC {auc:.4f}  AP {ap_:.4f}  ({time.time()-t0:.0f}s){flag}")
        if args.patience > 0 and epoch - best["epoch"] > args.patience:
            print("early stopping"); break

    model.load_state_dict(best["state"])
    torch.save(best["state"], osp.join(args.outdir, "best_model.pt"))
    print(f"best epoch {best['epoch']} (val AUROC {best['auc']:.4f})")

    va_probs, va_y = predict(model, ds_va, dev, args.eval_repeats, 1000)
    te_probs, te_y = predict(model, ds_te, dev, args.eval_repeats, 2000)
    thr, f1v = best_f1_threshold(va_probs[:, 1], va_y)

    idx = pd.read_csv(osp.join(args.dataset, "split_index.csv"))
    out = {}
    for name, probs, y in (("val", va_probs, va_y), ("test", te_probs, te_y)):
        rows = idx[idx["split"] == name].reset_index(drop=True)
        assert len(rows) == len(y) and (rows["label"].values == y).all(), "row order mismatch"
        out[name] = pd.DataFrame({"row_id": rows["row_id"], "actual": y, "prob_fraud": probs[:, 1],
                                  "pred_argmax": probs.argmax(1), "pred_valthr": (probs[:, 1] >= thr).astype(int)})
        out[name].to_csv(osp.join(args.outdir, f"predictions_{name}.csv"), index=False)

    te_auc, te_ap = score(te_probs, te_y)
    summary = {"best_epoch": best["epoch"], "val_auroc": best["auc"], "val_threshold_f1": thr, "val_f1_at_threshold": f1v,
               "test_auroc": te_auc, "test_ap": te_ap, "args": vars(args), "history": history}
    json.dump(summary, open(osp.join(args.outdir, "run_summary.json"), "w"), indent=2, default=str)
    print(f"test AUROC {te_auc:.4f}  AP {te_ap:.4f}  | threshold from val: {thr:.3f}")
    print("wrote", osp.join(args.outdir, "predictions_test.csv"))


if __name__ == "__main__":
    main()
