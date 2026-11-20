from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .recsys import score_from_ratings

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
RESULTS = ROOT / "results" / "results.json"
app = FastAPI(title="CinePredict · Collaborative Filtering API", version="2.0.0",
              description="MovieLens collaborative-filtering research demo based on the accompanying IEEE mini-project paper.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_origin_regex=r"http://172\.\d+\.\d+\.\d+:5173", allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

D = {}

def load_artifacts():
    p = np.load(MODEL / "params.npz")
    D.update(S=sp.load_npz(MODEL / "item_sim.npz").tocsr(),
             S_rank=sp.load_npz(MODEL / "item_sim_rank.npz").tocsr(),
             R=sp.load_npz(MODEL / "ratings.npz").tocsr(),
             mu=float(p["mu"]), bi=p["bi"].astype(np.float32), damp=float(p["damp"]),
             users=p["users"], items=p["items"], movies=pd.read_csv(MODEL / "movies_meta.csv"),
             report=json.loads(RESULTS.read_text()))
    D["movie_to_idx"] = {int(mid): i for i, mid in enumerate(D["items"])}
    D["user_to_idx"] = {int(uid): i for i, uid in enumerate(D["users"])}

@app.on_event("startup")
def startup():
    load_artifacts()

class RatingInput(BaseModel):
    movie_id: int
    rating: float = Field(ge=0.5, le=5.0)

class NewUserInput(BaseModel):
    ratings: list[RatingInput] = Field(min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=30)
    min_ratings: int = Field(default=5, ge=0, le=1000)


def movie_payload(indices, scores=None, estimates=None):
    out=[]
    for n, ix in enumerate(indices):
        row=D["movies"].iloc[int(ix)]
        item={"movie_id":int(row.movieId),"title":str(row.title),"genres":str(row.genres).replace("|", " · "),
              "rating_count":int(row.n_ratings),"average_rating":float(row.avg_rating)}
        if scores is not None: item["similarity_score"]=round(float(scores[n]),4)
        if estimates is not None: item["estimated_rating"]=round(float(estimates[int(ix)]),2)
        out.append(item)
    return out

def get_user_vec(uid):
    if uid not in D["user_to_idx"]: raise HTTPException(404, "MovieLens user ID not found")
    return D["R"][D["user_to_idx"][uid]].toarray().ravel()

def recommend_vec(vec, limit=10, min_ratings=5):
    est, rank=score_from_ratings(D["S"],D["S_rank"],D["mu"],D["bi"],vec,D["damp"])
    rank=rank.astype(float).copy(); rank[vec>0]=-np.inf
    rank[D["movies"].n_ratings.to_numpy()<min_ratings]=-np.inf
    top=np.argsort(-rank)[:limit]
    return movie_payload(top, scores=rank[top], estimates=est)

@app.get("/")
def root(): return {"name":"CinePredict Collaborative Filtering API","docs":"/docs","health":"/api/health"}

@app.get("/api/health")
def health(): return {"status":"ok","model_loaded":bool(D),"users":len(D.get("users",[])),"movies":len(D.get("items",[]))}

@app.get("/api/overview")
def overview():
    e=D["report"]["eda"]
    return {"dataset":"MovieLens ml-latest-small","ratings":e["n_ratings"],"users":e["n_users"],"movies_rated":e["n_movies_rated"],
            "catalogue_movies":e["n_movies_catalogue"],"sparsity":e["sparsity"],"mean_rating":e["mean_rating"],
            "median_ratings_per_user":e["median_ratings_per_user"],"median_ratings_per_movie":e["median_ratings_per_movie"],
            "best_model":"Item-based KNN","best_rmse":0.854,"best_mae":0.653,"best_ndcg10":0.245}

@app.get("/api/evaluation")
def evaluation():
    p=ROOT/"results"/"paper_metrics.json"
    return json.loads(p.read_text())

@app.get("/api/users")
def users(q: str = Query(default="",max_length=40), limit:int=Query(default=30,ge=1,le=610)):
    ids=D["users"].tolist()
    if q.strip():
        ids=[u for u in ids if q.strip() in str(u)]
    return [{"user_id":int(u),"ratings_count":int(D["R"][D["user_to_idx"][int(u)]].getnnz())} for u in ids[:limit]]

@app.get("/api/movies")
def movies(q:str=Query(default="",max_length=120),limit:int=Query(default=30,ge=1,le=100)):
    df=D["movies"]
    if q.strip(): df=df[df.title.str.contains(q.strip(),case=False,regex=False)]
    return movie_payload(df.index[:limit].tolist())

@app.get("/api/recommendations/user/{user_id}")
def user_recommendations(user_id:int,limit:int=Query(default=10,ge=1,le=30),min_ratings:int=Query(default=5,ge=0,le=1000)):
    vec=get_user_vec(user_id)
    liked=np.flatnonzero(vec>0)
    top_liked=liked[np.argsort(-vec[liked])[:8]]
    return {"user_id":user_id,"ratings_count":int(len(liked)),"top_rated":movie_payload(top_liked,estimates=vec),
            "recommendations":recommend_vec(vec,limit,min_ratings)}

@app.get("/api/similar/{movie_id}")
def similar_movies(movie_id:int,limit:int=Query(default=10,ge=1,le=30),min_ratings:int=Query(default=5,ge=0,le=1000)):
    if movie_id not in D["movie_to_idx"]: raise HTTPException(404,"Movie ID not found in the rated catalogue")
    ix=D["movie_to_idx"][movie_id]
    row=D["S"][ix].toarray().ravel().copy(); row[ix]=-np.inf
    row[row <= 0] = -np.inf
    row[D["movies"].n_ratings.to_numpy()<min_ratings]=-np.inf
    top=np.argsort(-row)[:limit]
    return {"source":movie_payload([ix])[0],"similar":movie_payload(top,scores=row[top])}

@app.post("/api/recommendations/new-user")
def new_user(payload:NewUserInput):
    vec=np.zeros(len(D["items"]),dtype=np.float32)
    for r in payload.ratings:
        ix=D["movie_to_idx"].get(r.movie_id)
        if ix is None: raise HTTPException(404,f"Movie ID {r.movie_id} is not in the MovieLens catalogue")
        vec[ix]=r.rating
    return {"ratings_count":int((vec>0).sum()),"recommendations":recommend_vec(vec,payload.limit,payload.min_ratings)}
