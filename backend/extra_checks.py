"""Ablations explaining the top-N results: (1) scoring rule, (2) minimum-support filter on candidate items."""
import json, sys
import numpy as np
sys.path.insert(0, "src")
from recsys import *

ratings, movies = load_data("data"); idx = Index(ratings)
best = json.load(open("results/results.json"))["best"]
bi, bs = best["itemknn"], best["svd"]
rows = {}
for seed in (42, 1, 2):
    tr, te = per_user_split(ratings, 0.2, seed=seed); R = to_matrix(idx, tr)
    support = (R > 0).sum(0)
    ik = ItemKNN(k=bi["k"], shrink=bi["shrink"], damp=bi["damp"], k_rank=best["itemknn_rank"]["k_rank"]).fit(R)
    svd = SVDMF(n_factors=bs["factors"], reg=bs["reg"], epochs=20).fit(R)
    variants = {
        "Item-KNN, similarity-sum score": ik.rank_all(),
        "Item-KNN, predicted-rating score": ik.predict_all(),
        "SVD, predicted-rating score": svd.predict_all(),
    }
    for name, sc in list(variants.items()):
        sc = sc.astype(np.float64)
        variants[name + " + min 20 ratings"] = np.where(support[None, :] >= 20, sc, -1e9)
    for name, sc in variants.items():
        m = topn_metrics(sc, R, idx, te, K=10)
        rows.setdefault(name, []).append(m)
out = {n: {k: [float(np.mean([r[k] for r in v])), float(np.std([r[k] for r in v]))] for k in v[0] if k != "users_evaluated"} for n, v in rows.items()}
json.dump(out, open("results/extra.json", "w"), indent=2)
for n, v in out.items(): print(f"{n:55s} P@10={v['Precision@10'][0]:.4f} NDCG@10={v['NDCG@10'][0]:.4f} Hit={v['HitRate@10'][0]:.4f}")
