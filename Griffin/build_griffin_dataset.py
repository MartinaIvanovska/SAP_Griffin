#!/usr/bin/env python

import argparse, hashlib, json, os, os.path as osp
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET = "creditcard"
TBL = f"{DATASET}-Transactions"
VOC = f"{TBL}_Class"
TASK = f"{TBL}-Class"
TASK_DESC = "Classify whether a credit card transaction is fraudulent based on its transaction data."
INT64_MIN = -9223372036854775808
DIM = 512


class Embedder:

    def __init__(self, fake: bool, device: str, cache: str):
        self.fake = fake
        if fake:
            print("!! --fake-embeddings: random name embeddings. ONLY for a smoke test, never for results.")
            return
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", device=device, cache_folder=cache,
            trust_remote_code=True, truncate_dim=DIM,
        )

    def __call__(self, texts):
        if self.fake:
            out = []
            for t in texts:
                seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
                out.append(np.random.RandomState(seed).randn(DIM))
            e = np.asarray(out, dtype="float32")
        else:
            e = self.model.encode(list(texts), batch_size=16, prompt="clustering: ").astype("float32")
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data_creditcard_with_class_more_testcases.csv")
    p.add_argument("--out", default="datasets/creditcard-griffin")
    p.add_argument("--label-col", default="Class")
    p.add_argument("--truth-col", default="ActualClass")
    p.add_argument("--val-frac", type=float, default=0.15, help="share of the 1850 context rows used for validation / early stopping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed-device", default="cpu")
    p.add_argument("--cache", default="cache_data/model")
    p.add_argument("--fake-embeddings", action="store_true")
    p.add_argument("--label-texts", nargs=2, default=["normal transaction", "fraud"],
                   help="words used in 'The Class is <word>' for class 0 / class 1")
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    df.insert(0, "row_id", np.arange(len(df)))
    is_test = df[a.label_col].astype(str).str.strip() == "[PREDICT]"
    ctx, test = df[~is_test].copy(), df[is_test].copy()
    ctx[a.label_col] = ctx[a.label_col].astype(int)
    test_truth = test[a.truth_col].astype(int).values

    skip = {"row_id", a.label_col, a.truth_col}
    feats = [c for c in df.columns if c not in skip]
    for c in feats:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise SystemExit(f"column {c!r} is not numeric; this builder expects the numeric ULB-style features")

    train, val = train_test_split(ctx, test_size=a.val_frac, stratify=ctx[a.label_col], random_state=a.seed)
    train = train.sample(frac=1.0, random_state=a.seed)        # random order inside each split so
    val = val.sample(frac=1.0, random_state=a.seed)            # frauds are not clustered
    ordered = pd.concat([train, val, test], ignore_index=True)
    n_tr, n_val, n_te = len(train), len(val), len(test)
    N = len(ordered)
    print(f"rows: train={n_tr} (fraud {int(train[a.label_col].sum())})  "
          f"val={n_val} (fraud {int(val[a.label_col].sum())})  "
          f"test={n_te} (fraud {int(test_truth.sum())})")

    split = np.array(["train"] * n_tr + ["val"] * n_val + ["test"] * n_te)
    label_true = np.concatenate([train[a.label_col].values, val[a.label_col].values, test_truth]).astype(np.int64)

    scaler = StandardScaler().fit(ordered.loc[:n_tr - 1, feats].fillna(ordered.loc[:n_tr - 1, feats].mean()))
    X = scaler.transform(ordered[feats].fillna(ordered.loc[:n_tr - 1, feats].mean())).astype("float32")

    text_idx = label_true.copy()
    text_idx[split == "test"] = 2

    os.makedirs(a.out, exist_ok=True)
    emb = Embedder(a.fake_embeddings, a.embed_device, a.cache)
    t0, t1 = a.label_texts
    label_text = [f"The Class is {t0}", f"The Class is {t1}", "The Class is unknown"]
    label_emb = emb(label_text)                                 # (3, 512)

    f_text = f"{TBL}___Griffin_text_Class"
    feat_names = [f"{TBL}___{c}" for c in feats] + [f_text]
    cols = {"timestamp": np.full(N, INT64_MIN, dtype=np.int64)}
    for i, c in enumerate(feats):
        cols[f"{TBL}___{c}"] = X[:, i]
    cols[f_text] = text_idx.astype(np.int64)
    Dataset.from_dict(cols).save_to_disk(osp.join(a.out, "node", TBL, "feat"))
    Dataset.from_dict({"emb": label_emb}).save_to_disk(osp.join(a.out, "node", TBL, "textemb"))

    v_text = f"{VOC}___Griffin_text_Class"
    Dataset.from_dict({"timestamp": np.full(2, INT64_MIN, dtype=np.int64),
                       v_text: np.array([0, 1], dtype=np.int64)}).save_to_disk(osp.join(a.out, "node", VOC, "feat"))
    Dataset.from_dict({"emb": label_emb[:2]}).save_to_disk(osp.join(a.out, "node", VOC, "textemb"))

    with open(osp.join(a.out, "metanode.yaml"), "w") as f:
        yaml.dump({TBL: {"num": int(N), "feat": feat_names, "is_target": False},
                   VOC: {"num": 2, "feat": [v_text], "is_target": True}}, f)
    with open(osp.join(a.out, "metaadj.yaml"), "w") as f:      # single table: no edges
        yaml.dump({TBL: {"out": [], "in": []}, VOC: {"out": [], "in": []}}, f)

    col_text = {f"{TBL}___{c}": f"table: {TBL}, column: {c}" for c in feats}
    col_text[f_text] = f"table: {TBL}, column: Class"
    col_text[v_text] = f"table: {VOC}, column: Class"
    keys = list(col_text)
    E = emb([col_text[k] for k in keys])
    torch.save({k: torch.from_numpy(E[i]).to(torch.float32) for i, k in enumerate(keys)},
               osp.join(a.out, "featnameemb.pt"))
    torch.save({"fewshot": torch.from_numpy(emb(["fewshot"])[0]).to(torch.float32)},
               osp.join(a.out, "edgenameemb.pt"))
    torch.save({TASK: torch.from_numpy(emb([TASK_DESC])[0]).to(torch.float32)},
               osp.join(a.out, "tasknameemb.pt"))

    Dataset.from_dict({"nodeidx": np.arange(N, dtype=np.int64), "label": label_true}).save_to_disk(
        osp.join(a.out, "task", TASK))
    with open(osp.join(a.out, "metatask.yaml"), "w") as f:
        yaml.dump({TASK: {
            "name": TASK, "task_type": "retrieval", "num_class": 2,
            "seed_type": VOC, "target_type": TBL,
            "split": [int(n_tr), int(n_val), int(n_te)],
            "metric": "retrieval_auroc", "hastimestamp": False,
            "extra_feat": [], "masked_feat": [f_text],
        }}, f)

    pd.DataFrame({"node_idx": np.arange(N), "row_id": ordered["row_id"].values,
                  "split": split, "label": label_true}).to_csv(osp.join(a.out, "split_index.csv"), index=False)
    json.dump({"features": feats, "label_texts": label_text, "seed": a.seed, "fake_embeddings": a.fake_embeddings,
               "rows": {"train": n_tr, "val": n_val, "test": n_te}},
              open(osp.join(a.out, "build_info.json"), "w"), indent=2)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
