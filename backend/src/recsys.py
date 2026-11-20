"""Collaborative-filtering recommenders on MovieLens ml-latest-small (numpy/pandas only)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 42


def load_data(data_dir="data"):
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    movies = pd.read_csv(f"{data_dir}/movies.csv")
    return ratings, movies


def per_user_split(df, test_frac=0.2, seed=SEED):
    """Hold out test_frac of EACH user's ratings (every user keeps training data)."""
    rng = np.random.RandomState(seed)
    test_idx = []
    for _, g in df.groupby("userId"):
        n_test = max(1, int(round(len(g) * test_frac)))
        test_idx.extend(rng.choice(g.index.values, n_test, replace=False))
    test = df.loc[test_idx]
    return df.drop(test.index), test


class Index:
    """Maps raw userId/movieId to dense matrix indices (built on ALL users/movies)."""
    def __init__(self, ratings):
        self.users = np.sort(ratings.userId.unique())
        self.items = np.sort(ratings.movieId.unique())
        self.u2i = {u: i for i, u in enumerate(self.users)}
        self.m2i = {m: i for i, m in enumerate(self.items)}

    def encode(self, df):
        return (df.userId.map(self.u2i).values, df.movieId.map(self.m2i).values,
                df.rating.values.astype(np.float32))


def to_matrix(idx, df):
    u, i, r = idx.encode(df)
    R = np.zeros((len(idx.users), len(idx.items)), dtype=np.float32)
    R[u, i] = r
    return R


class Baseline:
    """Global mean + regularised item bias + regularised user bias."""
    def __init__(self, reg_i=10.0, reg_u=15.0):
        self.reg_i, self.reg_u = reg_i, reg_u

    def fit(self, R):
        M = R > 0
        self.mu = R[M].mean()
        self.bi = ((R - self.mu) * M).sum(0) / (self.reg_i + M.sum(0))
        resid = (R - self.mu - self.bi) * M
        self.bu = resid.sum(1) / (self.reg_u + M.sum(1))
        return self

    def predict_all(self):
        return self.mu + self.bu[:, None] + self.bi[None, :]


def _topk_sparsify(S, k):
    """Keep the k largest-|similarity| entries per row (dense in -> dense out; small matrices only)."""
    keep = np.argpartition(-np.abs(S), k, axis=1)[:, :k]
    T = np.zeros_like(S)
    rows = np.arange(S.shape[0])[:, None]
    T[rows, keep] = S[rows, keep]
    return T


def _topk_rows_to_csr(S, k, chunk=1000):
    """Memory-safe top-k per row for a large dense matrix; returns a CSR matrix."""
    n = S.shape[0]
    indptr, indices, data = [0], [], []
    for a in range(0, n, chunk):
        blk = S[a:a + chunk]
        keep = np.argpartition(-np.abs(blk), k, axis=1)[:, :k]
        for r in range(blk.shape[0]):
            cols = np.sort(keep[r])
            indices.append(cols); data.append(blk[r, cols])
            indptr.append(indptr[-1] + len(cols))
    return sp.csr_matrix((np.concatenate(data).astype(np.float32), np.concatenate(indices), np.array(indptr)),
                         shape=S.shape)


class ItemKNN:
    """Item-based CF: residual cosine similarity + shrinkage, top-k neighbours per item (sparse).
    Rating prediction : baseline + damped weighted average of neighbour residuals.
    Top-N ranking     : sum of similarities to the items the user liked (rating >= 4)."""
    def __init__(self, k=40, shrink=25.0, damp=0.0, k_rank=None):
        self.k, self.shrink, self.damp, self.k_rank = k, shrink, damp, k_rank

    def fit(self, R, keep_full=False, chunk=1000):
        self.base = Baseline().fit(R)
        M = (R > 0).astype(np.float32)
        X = ((R - self.base.predict_all()) * M).astype(np.float32)  # residuals, 0 where unrated
        norms = (np.sqrt((X ** 2).sum(0)) + 1e-8).astype(np.float32)
        ni = R.shape[1]
        full = np.zeros((ni, ni), dtype=np.float32)
        for a in range(0, ni, chunk):
            b = min(a + chunk, ni)
            s = (X[:, a:b].T @ X) / np.outer(norms[a:b], norms)    # cosine of residual vectors
            c = M[:, a:b].T @ M
            s *= c / (c + self.shrink)                              # shrink low-overlap similarities
            s[np.arange(b - a), np.arange(a, b)] = 0                # no self-similarity
            full[a:b] = s
        self._S_full = full if keep_full else None
        self.X, self.M, self.L = X, M, (R >= 4.0).astype(np.float32)
        self._sparsify(full)
        return self

    def _sparsify(self, full):
        self.S = _topk_rows_to_csr(full, self.k)
        kr = self.k_rank or self.k
        self.S_rank = self.S if kr == self.k else _topk_rows_to_csr(full, kr)

    def set_k(self, k, k_rank=None):
        """Re-sparsify the cached full similarity matrix (needs fit(..., keep_full=True))."""
        self.k, self.k_rank = k, k_rank
        self._sparsify(self._S_full)
        return self

    def predict_all(self):
        num = (self.S @ self.X.T).T
        den = (abs(self.S) @ self.M.T).T + self.damp + 1e-8
        return np.clip(self.base.predict_all() + num / den, 0.5, 5.0)

    def rank_all(self):
        return (self.S_rank @ self.L.T).T

    def similar_items(self, item_idx, n=10):
        row = self.S[item_idx].toarray().ravel()
        order = np.argsort(-row)[:n]
        return order, row[order]


class UserKNN:
    """User-based CF: residual cosine similarity between users, top-k neighbours."""
    def __init__(self, k=40, shrink=10.0, damp=0.0, k_rank=None):
        self.k, self.shrink, self.damp, self.k_rank = k, shrink, damp, k_rank

    def fit(self, R, keep_full=False):
        self.base = Baseline().fit(R)
        M = (R > 0).astype(np.float32)
        X = (R - self.base.predict_all()) * M
        norms = np.sqrt((X ** 2).sum(1)) + 1e-8
        S = (X @ X.T) / np.outer(norms, norms)
        C = M @ M.T
        S *= C / (C + self.shrink)
        np.fill_diagonal(S, 0)
        self._S_full = S
        self.X, self.M, self.L = X, M, (R >= 4.0).astype(np.float32)
        self.set_k(self.k, self.k_rank)
        return self

    def set_k(self, k, k_rank=None):
        self.k, self.k_rank = k, k_rank
        self.S = _topk_sparsify(self._S_full, k)
        self.S_rank = self.S if not k_rank else _topk_sparsify(self._S_full, k_rank)
        return self

    def predict_all(self):
        num = self.S @ self.X
        den = np.abs(self.S) @ self.M + self.damp + 1e-8
        return np.clip(self.base.predict_all() + num / den, 0.5, 5.0)

    def rank_all(self):
        return self.S_rank @ self.L


class SVDMF:
    """Funk-SVD matrix factorisation (user/item biases + latent factors) trained with SGD."""
    def __init__(self, n_factors=30, lr=0.01, reg=0.05, epochs=25, seed=SEED):
        self.f, self.lr, self.reg, self.epochs, self.seed = n_factors, lr, reg, epochs, seed

    def fit(self, R):
        rng = np.random.RandomState(self.seed)
        u_idx, i_idx = np.nonzero(R)
        r = R[u_idx, i_idx]
        nu, ni = R.shape
        self.mu = float(r.mean())
        self.bu, self.bi = np.zeros(nu), np.zeros(ni)
        self.P = rng.normal(0, 0.1, (nu, self.f))
        self.Q = rng.normal(0, 0.1, (ni, self.f))
        lr, reg = self.lr, self.reg
        for _ in range(self.epochs):
            for n in rng.permutation(len(r)):
                u, i = u_idx[n], i_idx[n]
                err = r[n] - (self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])
                self.bu[u] += lr * (err - reg * self.bu[u])
                self.bi[i] += lr * (err - reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += lr * (err * self.Q[i] - reg * pu)
                self.Q[i] += lr * (err * pu - reg * self.Q[i])
        return self

    def predict_all(self):
        full = self.mu + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T
        return np.clip(full, 0.5, 5.0)


class Popularity:
    """Non-personalised baseline: damped (Bayesian) mean rating per movie."""
    def fit(self, R, m=20):
        M = R > 0
        mu = R[M].mean()
        self.score = (R.sum(0) + m * mu) / (M.sum(0) + m)
        self.nu = R.shape[0]
        return self

    def predict_all(self):
        return np.tile(self.score, (self.nu, 1))


class PopularityCount:
    """Classic non-personalised top-N baseline: rank movies by number of ratings."""
    rating_ok = False
    def fit(self, R):
        self.score, self.nu = (R > 0).sum(0).astype(np.float32), R.shape[0]
        return self

    def predict_all(self):
        return np.tile(self.score, (self.nu, 1))


class GlobalMean:
    def fit(self, R):
        self.mu, self.shape = R[R > 0].mean(), R.shape
        return self

    def predict_all(self):
        return np.full(self.shape, self.mu, dtype=np.float32)


# ------------------------------------------------------------------ evaluation
def rating_metrics(pred, idx, test):
    u, i, r = idx.encode(test)
    p = pred[u, i]
    return {"RMSE": float(np.sqrt(np.mean((p - r) ** 2))), "MAE": float(np.mean(np.abs(p - r)))}


def topn_metrics(pred, R_train, idx, test, K=10, thresh=4.0):
    """Rank all unseen movies per user; 'relevant' = held-out ratings >= thresh."""
    u, i, r = idx.encode(test)
    rel = {}
    for uu, ii, rr in zip(u, i, r):
        if rr >= thresh:
            rel.setdefault(uu, set()).add(ii)
    P, Rc, H, N = [], [], [], []
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    for uu, relset in rel.items():
        s = pred[uu].astype(np.float64).copy()
        s[R_train[uu] > 0] = -np.inf                      # never recommend already-rated movies
        top = np.argpartition(-s, K)[:K]
        top = top[np.argsort(-s[top])]
        hits = np.array([t in relset for t in top], dtype=float)
        P.append(hits.sum() / K)
        Rc.append(hits.sum() / len(relset))
        H.append(float(hits.sum() > 0))
        N.append((hits * disc).sum() / disc[:min(len(relset), K)].sum())
    return {f"Precision@{K}": float(np.mean(P)), f"Recall@{K}": float(np.mean(Rc)),
            f"NDCG@{K}": float(np.mean(N)), f"HitRate@{K}": float(np.mean(H)),
            "users_evaluated": len(rel)}


def catalog_coverage(pred, R_train, K=10):
    seen = set()
    for uu in range(pred.shape[0]):
        s = pred[uu].astype(np.float64).copy()
        s[R_train[uu] > 0] = -np.inf
        seen.update(np.argpartition(-s, K)[:K].tolist())
    return len(seen) / pred.shape[1]


# ------------------------------------------------- serving (used by the demo app)
def score_from_ratings(S_rating, S_rank, mu, bi, ratings_vec, damp=0.0, reg_u=15.0):
    """Item-based CF for ONE user from a dense rating vector (0 = unrated). Works for existing and
    brand-new users alike. Returns (estimated_rating, ranking_score) over all movies."""
    mask = (ratings_vec > 0).astype(np.float32)
    n = mask.sum()
    bu = ((ratings_vec - mu - bi) * mask).sum() / (reg_u + n) if n > 0 else 0.0
    x = (ratings_vec - mu - bi - bu) * mask
    est = np.clip(mu + bi + bu + (S_rating @ x) / (abs(S_rating) @ mask + damp + 1e-8), 0.5, 5.0)
    rank = S_rank @ (ratings_vec >= 4.0).astype(np.float32)
    return est, rank
