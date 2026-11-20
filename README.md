# 🎬 CinePredict — Collaborative Filtering for Movie Recommendation

A comparative study of neighbourhood-based and latent-factor methods for movie recommendation using the MovieLens dataset.

CinePredict explores how collaborative filtering algorithms predict user ratings and generate personalized Top-10 movie recommendations. The project compares traditional baselines, user-based and item-based KNN, and matrix factorization under a consistent, leakage-free evaluation protocol.

---

## 📌 Overview

Movie recommendation systems help users discover movies based on their preferences and the preferences of similar users.

This project investigates two important tasks:

- **Rating Prediction:** Predict how a user might rate a movie.
- **Top-N Recommendation:** Recommend the most relevant movies a user hasn't watched.

We compare seven methods and investigate whether better rating prediction necessarily leads to better movie recommendations.

Our experiments show that **Item-Based Collaborative Filtering with residual cosine similarity and shrinkage** performs best across both rating prediction and Top-10 recommendation ranking in our experimental setup.

---

## 🎯 Objectives

- Compare personalized collaborative filtering with non-personalized baselines.
- Evaluate user-based KNN and item-based KNN.
- Compare neighborhood methods with matrix factorization using SVD.
- Evaluate rating prediction and recommendation ranking together.
- Study the effect of different ranking/scoring rules.
- Maintain a reproducible and leakage
