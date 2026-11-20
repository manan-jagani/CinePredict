"""Run the full pipeline: EDA -> tuning (validation only) -> final test evaluation (3 splits) -> figures -> demo artifacts.
Usage:  python run_experiments.py        (needs data/ratings.csv and data/movies.csv; ~5-10 min on a laptop)
"""
import json, os, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp

sys.path.insert(0, "src")
from recsys import *

os.makedirs("results/figures", exist_ok=True)
os.makedirs("model", exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#2f5597", "#c55a11"

ratings, movies = load_data("data")
idx = Index(ratings)
nu, ni = len(idx.users), len(idx.items)

# ------------------------------------------------------------------ 1. EDA
per_user, per_movie = ratings.groupby("userId").size(), ratings.groupby("movieId").size()
eda = {"n_ratings": len(ratings), "n_users": nu, "n_movies_rated": ni, "n_movies_catalogue": len(movies),
       "sparsity": 1 - len(ratings) / (nu * ni), "mean_rating": float(ratings.rating.mean()),
       "median_ratings_per_user": float(per_user.median()), "min_ratings_per_user": int(per_user.min()),
       "median_ratings_per_movie": float(per_movie.median()), "movies_with_le5_ratings_pct": float((per_movie <= 5).mean() * 100)}
log("EDA", eda)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0))
ratings.rating.value_counts().sort_index().plot.bar(ax=ax[0], color=BLUE); ax[0].set(xlabel="Rating", ylabel="Count")
per_user.plot.hist(bins=40, ax=ax[1], color=BLUE, log=True); ax[1].set(xlabel="Ratings per user", ylabel="Users (log)")
per_movie.plot.hist(bins=40, ax=ax[2], color=BLUE, log=True); ax[2].set(xlabel="Ratings per movie", ylabel="Movies (log)")
plt.tight_layout(); plt.savefig("results/figures/eda.png"); plt.close()

# ------------------------------------------------------------------ 2. tuning on a validation split (TRAIN only)
train42, _ = per_user_split(ratings, 0.2, seed=42)
tr_sub, val = per_user_split(train42, 0.1, seed=7)
R_sub = to_matrix(idx, tr_sub)
tuning = {"itemknn": [], "userknn": [], "svd": [], "itemknn_rank": [], "userknn_rank": []}

for shrink in (10.0, 25.0, 50.0, 100.0):
    m = ItemKNN(k=40, shrink=shrink).fit(R_sub, keep_full=True)
    for k in (40, 80, 160, 320, 640):
        m.set_k(k)
        for damp in (0.0, 2.0, 5.0):
            m.damp = damp
            tuning["itemknn"].append({"k": k, "shrink": shrink, "damp": damp, **rating_metrics(m.predict_all(), idx, val)})
    log("itemknn shrink", shrink, "done")
    del m
bi_ = min(tuning["itemknn"], key=lambda d: d["RMSE"])
for k in (10, 20, 40, 80, 160, 320):
    for damp in (0.0, 2.0, 5.0):
        r = rating_metrics(UserKNN(k=k, shrink=10.0, damp=damp).fit(R_sub).predict_all(), idx, val)
        tuning["userknn"].append({"k": k, "shrink": 10.0, "damp": damp, **r})
bu_ = min(tuning["userknn"], key=lambda d: d["RMSE"])
for f, reg in ((20, 0.05), (30, 0.1), (60, 0.1)):
    r = rating_metrics(SVDMF(n_factors=f, reg=reg, epochs=20).fit(R_sub).predict_all(), idx, val)
    tuning["svd"].append({"factors": f, "reg": reg, **r}); log("svd", f, reg, r)
bs_ = min(tuning["svd"], key=lambda d: d["RMSE"])

# ranking-oriented tuning: neighbourhood size for similarity-sum scoring, chosen by validation NDCG@10
m = ItemKNN(k=bi_["k"], shrink=bi_["shrink"]).fit(R_sub, keep_full=True)
for kr in (20, 50, 100, 200, 400):
    m.set_k(bi_["k"], kr)
    tuning["itemknn_rank"].append({"k_rank": kr, **topn_metrics(m.rank_all(), R_sub, idx, val)})
del m
mu_ = UserKNN(k=bu_["k"], shrink=10.0).fit(R_sub)
for kr in (10, 20, 40, 80, 160):
    mu_.set_k(bu_["k"], kr)
    tuning["userknn_rank"].append({"k_rank": kr, **topn_metrics(mu_.rank_all(), R_sub, idx, val)})
bir_ = max(tuning["itemknn_rank"], key=lambda d: d["NDCG@10"])
bur_ = max(tuning["userknn_rank"], key=lambda d: d["NDCG@10"])
best = {"itemknn": bi_, "userknn": bu_, "svd": bs_, "itemknn_rank": bir_, "userknn_rank": bur_}
log("BEST", json.dumps(best, default=float))

# ------------------------------------------------------------------ 3. final evaluation (3 random 80/20 splits)
def make_models():
    return {
        "Global mean": GlobalMean(),
        "Popularity (rating count)": PopularityCount(),
        "Popularity (damped mean)": Popularity(),
        "Bias baseline": Baseline(),
        "User-based KNN": UserKNN(k=bu_["k"], shrink=10.0, damp=bu_["damp"], k_rank=bur_["k_rank"]),
        "Item-based KNN": ItemKNN(k=bi_["k"], shrink=bi_["shrink"], damp=bi_["damp"], k_rank=bir_["k_rank"]),
        "SVD (matrix factorisation)": SVDMF(n_factors=bs_["factors"], reg=bs_["reg"], epochs=20),
    }

all_runs = {}
for seed in (42, 1, 2):
    tr, te = per_user_split(ratings, 0.2, seed=seed)
    R = to_matrix(idx, tr)
    for name, model in make_models().items():
        model.fit(R)
        pred = model.predict_all()
        res = rating_metrics(pred, idx, te) if getattr(model, "rating_ok", True) else {}
        if name != "Global mean":
            scores = model.rank_all() if hasattr(model, "rank_all") else pred
            res.update(topn_metrics(scores, R, idx, te, K=10, thresh=4.0))
            res["Coverage@10"] = catalog_coverage(scores, R, K=10)
        all_runs.setdefault(name, []).append(res)
        log(seed, name, {k: round(v, 4) for k, v in res.items()})
        del pred, model

summary = {n: {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
               for k in runs[0] if k != "users_evaluated"} for n, runs in all_runs.items()}

# ------------------------------------------------------------------ 4. figures
names = list(summary)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
for a, met, ttl in ((ax[0], "RMSE", "Test RMSE (lower is better)"), (ax[1], "NDCG@10", "Test NDCG@10 (higher is better)")):
    ns = [n for n in names if met in summary[n]]
    a.barh(ns, [summary[n][met]["mean"] for n in ns], xerr=[summary[n][met]["std"] for n in ns], color=BLUE, capsize=2)
    a.set_title(ttl, fontsize=8); a.invert_yaxis()
plt.tight_layout(); plt.savefig("results/figures/comparison.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
t = pd.DataFrame(tuning["itemknn"]); t = t[t.damp == bi_["damp"]]
for s_, g in t.groupby("shrink"): ax[0].plot(g.k, g.RMSE, marker="o", ms=3, label=f"shrink={int(s_)}")
ax[0].set(xlabel="Neighbours k", ylabel="Validation RMSE", title="Item-based KNN (rating prediction)", xscale="log"); ax[0].legend(fontsize=6)
for lbl, key, c in (("Item-based", "itemknn_rank", BLUE), ("User-based", "userknn_rank", ORANGE)):
    d = pd.DataFrame(tuning[key]); ax[1].plot(d.k_rank, d["NDCG@10"], marker="o", ms=3, color=c, label=lbl)
ax[1].set(xlabel="Neighbours k", ylabel="Validation NDCG@10", title="Similarity-sum ranking", xscale="log"); ax[1].legend(fontsize=6)
from matplotlib.ticker import ScalarFormatter
for a in ax:
    a.xaxis.set_major_formatter(ScalarFormatter()); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].set_xticks([40, 80, 160, 320, 640]); ax[1].set_xticks([10, 20, 50, 100, 200, 400])
plt.tight_layout(); plt.savefig("results/figures/k_sensitivity.png"); plt.close()

# ------------------------------------------------------------------ 5. deployment model on ALL ratings -> demo artifacts
log("fitting deployment model on all ratings")
R_all = to_matrix(idx, ratings)
final = ItemKNN(k=bi_["k"], shrink=bi_["shrink"], damp=bi_["damp"], k_rank=bir_["k_rank"]).fit(R_all)
sp.save_npz("model/item_sim.npz", final.S)
sp.save_npz("model/item_sim_rank.npz", final.S_rank)
sp.save_npz("model/ratings.npz", sp.csr_matrix(R_all))
np.savez("model/params.npz", mu=np.float32(final.base.mu), bi=final.base.bi.astype(np.float32),
         damp=np.float32(final.damp), users=idx.users, items=idx.items)
mv = movies.set_index("movieId").loc[idx.items].reset_index()
mv["n_ratings"] = mv.movieId.map(per_movie).values
mv["avg_rating"] = mv.movieId.map(ratings.groupby("movieId").rating.mean()).round(2).values
mv.to_csv("model/movies_meta.csv", index=False)

title = lambda i: mv.title.iloc[i]
ex = {}
toy = int(np.where(idx.items == 1)[0][0])
o, s_ = final.similar_items(toy, 5)
ex["similar_to_toy_story"] = [(title(i), round(float(v), 3)) for i, v in zip(o, s_)]
uu = idx.u2i[1]
est, rank = score_from_ratings(final.S, final.S_rank, final.base.mu, final.base.bi, R_all[uu].copy(), final.damp)
rank = rank.copy(); rank[R_all[uu] > 0] = -np.inf
ex["user1_top_rated"] = [title(i) for i in np.argsort(-R_all[uu])[:5]]
ex["user1_recommended"] = [(title(i), round(float(est[i]), 2)) for i in np.argsort(-rank)[:5]]
log("examples", ex)

json.dump({"eda": eda, "tuning": tuning, "best": best, "summary": summary, "runs": all_runs, "examples": ex},
          open("results/results.json", "w"), indent=2, default=float)
log("DONE")
