
import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Bookmark,
  Clapperboard,
  Compass,
  Film,
  Layers3,
  LoaderCircle,
  Mic2,
  Search,
  ShieldCheck,
  Sparkles,
  Star,
  TrendingUp,
  Users,
  X
} from 'lucide-react';

import './style.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    },
    ...options
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data.detail || 'The research service could not complete this request.'
    );
  }

  return data;
}

// MovieLens stores some titles with leading articles at the end, e.g.
// "Matrix, The (1999)". Format titles for display without changing API values.
function formatMovieTitle(title = '') {
  return String(title).replace(
    /^(.+),\s(The|A|An)(\s*\(\d{4}\))?$/,
    '$2 $1$3'
  );
}

const posterClasses = [
  'poster-a',
  'poster-b',
  'poster-c',
  'poster-d',
  'poster-e',
  'poster-f'
];

function Poster({ movie, index = 0 }) {
  return (
    <div className={`poster ${posterClasses[index % posterClasses.length]}`}>
      <div className="poster-grain" />
      <div className="poster-orbit" />

      <div className="poster-title">
        {formatMovieTitle(movie?.title).split(' (')[0] || 'CINEMA'}
        <span>
          {movie?.title?.match(/\((\d{4})\)/)?.[1] || 'FEATURE'}
        </span>
      </div>

      <div className="poster-foot">
        <Film size={17} />
        <i />
      </div>
    </div>
  );
}

function Rating({ value }) {
  return (
    <span className="rating">
      <Star size={13} fill="currentColor" />
      {Number(value || 0).toFixed(2)}
    </span>
  );
}

function MovieCard({
  movie,
  index = 0,
  tag,
  onSimilar,
  isSaved = false,
  onToggleSave
}) {
  return (
    <article className="movie-card">
      <div className="poster-wrap">
        <Poster movie={movie} index={index} />

        {tag && <span className="poster-tag">{tag}</span>}

        <button
          className={`save-btn ${isSaved ? 'saved' : ''}`}
          title={isSaved ? 'Remove from saved movies' : 'Save movie'}
          aria-label={isSaved ? 'Remove from saved movies' : 'Save movie'}
          aria-pressed={isSaved}
          type="button"
          onClick={() => onToggleSave?.(movie)}
        >
          <Bookmark size={14} fill={isSaved ? 'currentColor' : 'none'} />
        </button>
      </div>

      <div className="movie-info">
        <div className="movie-title-row">
          <h3>{formatMovieTitle(movie.title)}</h3>

          {movie.estimated_rating != null && (
            <strong className="pred-score">
              {Number(movie.estimated_rating).toFixed(1)}
              <small>/5</small>
            </strong>
          )}
        </div>

        <p>
          {movie.genres?.split(' · ').slice(0, 3).join(' · ')}
        </p>

        <div className="movie-meta">
          <Rating value={movie.average_rating} />
          <span>
            {Number(movie.rating_count || 0).toLocaleString()} ratings
          </span>
        </div>

        {onSimilar && (
          <button
            className="text-action"
            type="button"
            onClick={() => onSimilar(movie)}
          >
            Explore similar <ArrowUpRight size={13} />
          </button>
        )}
      </div>
    </article>
  );
}

function PageHeading({ eyebrow, title, desc }) {
  return (
    <div className="page-heading">
      <div className="eyebrow">
        <span />
        {eyebrow}
      </div>
      <h1>{title}</h1>
      <p>{desc}</p>
    </div>
  );
}

function App() {
  const [page, setPage] = useState('discover');
  const [savedMovies, setSavedMovies] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('cinepredict_saved_movies') || '[]');
    } catch {
      return [];
    }
  });
  const [health, setHealth] = useState(false);
  const [overview, setOverview] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const [userId, setUserId] = useState('1');
  const [userResult, setUserResult] = useState(null);

  const [movieQuery, setMovieQuery] = useState('Toy Story');
  const [movieResults, setMovieResults] = useState([]);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [similar, setSimilar] = useState(null);

  const [minRatings, setMinRatings] = useState(5);
  const [limit, setLimit] = useState(10);

  const [newQuery, setNewQuery] = useState('');
  const [newResults, setNewResults] = useState([]);
  const [newRatings, setNewRatings] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('cinepredict_taste_profile') || '[]');
    } catch {
      return [];
    }
  });
  const [newRecommendations, setNewRecommendations] = useState(null);

  // Persist bookmarks and the new-user taste profile across refreshes.
  useEffect(() => {
    localStorage.setItem('cinepredict_saved_movies', JSON.stringify(savedMovies));
  }, [savedMovies]);

  useEffect(() => {
    localStorage.setItem('cinepredict_taste_profile', JSON.stringify(newRatings));
  }, [newRatings]);

  function toggleSavedMovie(movie) {
    setSavedMovies(current => {
      const exists = current.some(
        item => String(item.movie_id) === String(movie.movie_id)
      );
      return exists
        ? current.filter(item => String(item.movie_id) !== String(movie.movie_id))
        : [...current, movie];
    });
  }

  function isMovieSaved(movie) {
    return savedMovies.some(
      item => String(item.movie_id) === String(movie.movie_id)
    );
  }

  // Initial API requests
  useEffect(() => {
    request('/api/health')
      .then(data => setHealth(data.model_loaded))
      .catch(() => setHealth(false));

    request('/api/overview')
      .then(setOverview)
      .catch(error => setError(error.message));

    request('/api/evaluation')
      .then(setEvaluation)
      .catch(() => {});
  }, []);

  // Similar movie search
  useEffect(() => {
    if (page !== 'similar' || movieQuery.trim().length <= 1) {
      setMovieResults([]);
      return;
    }

    let cancelled = false;

    const timeout = setTimeout(() => {
      request(
        `/api/movies?q=${encodeURIComponent(movieQuery)}&limit=12`
      )
        .then(data => {
          if (!cancelled) setMovieResults(data);
        })
        .catch(() => {
          if (!cancelled) setMovieResults([]);
        });
    }, 250);

    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [movieQuery, page]);

  // New user movie search
  useEffect(() => {
    if (page !== 'new' || newQuery.trim().length <= 1) {
      setNewResults([]);
      return;
    }

    let cancelled = false;

    const timeout = setTimeout(() => {
      request(
        `/api/movies?q=${encodeURIComponent(newQuery)}&limit=12`
      )
        .then(data => {
          if (!cancelled) setNewResults(data);
        })
        .catch(() => {
          if (!cancelled) setNewResults([]);
        });
    }, 250);

    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [newQuery, page]);

  async function loadUser(event) {
    event?.preventDefault();

    setBusy(true);
    setError('');

    try {
      const result = await request(
        `/api/recommendations/user/${userId}?limit=${limit}&min_ratings=${minRatings}`
      );

      setUserResult(result);
    } catch (error) {
      setError(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function loadSimilar(movie) {
    setSelectedMovie(movie);
    setBusy(true);
    setError('');

    try {
      const result = await request(
        `/api/similar/${movie.movie_id}?limit=${limit}&min_ratings=${minRatings}`
      );

      setSimilar(result);
    } catch (error) {
      setError(error.message);
    } finally {
      setBusy(false);
    }
  }

  function addRating(movie) {
    setNewRatings(old =>
      old.some(item => item.movie_id === movie.movie_id)
        ? old
        : [
            ...old,
            {
              movie_id: movie.movie_id,
              title: movie.title,
              rating: 4
            }
          ]
    );

    setNewQuery('');
    setNewResults([]);
  }

  async function getNewRecommendations() {
    setBusy(true);
    setError('');

    try {
      const result = await request('/api/recommendations/new-user', {
        method: 'POST',
        body: JSON.stringify({
          ratings: newRatings.map(({ movie_id, rating }) => ({
            movie_id,
            rating
          })),
          limit,
          min_ratings: minRatings
        })
      });

      setNewRecommendations(result);
    } catch (error) {
      setError(error.message);
    } finally {
      setBusy(false);
    }
  }

  const nav = [
    { id: 'discover', label: 'Discover', icon: Compass },
    { id: 'existing', label: 'For a MovieLens user', icon: Users },
    { id: 'similar', label: 'Similar movies', icon: Layers3 },
    { id: 'new', label: 'Your taste profile', icon: Sparkles },
    { id: 'saved', label: 'Saved movies', icon: Bookmark },
    { id: 'research', label: 'Research & metrics', icon: BarChart3 }
  ];

  const pageTitle = {
    discover: ['THE COLLECTION', 'Find your next favourite.'],
    existing: [
      'PERSONALISED DISCOVERY',
      'Recommendations shaped by a viewer.'
    ],
    similar: [
      'ITEM-TO-ITEM SIMILARITY',
      'One film. A universe of connections.'
    ],
    new: [
      'COLD-START ONBOARDING',
      'Teach the model what you love.'
    ],
    saved: [
      'YOUR COLLECTION',
      'The films you saved for later.'
    ],
    research: [
      'THE RESEARCH DESK',
      'Evidence behind every recommendation.'
    ]
  }[page];

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Clapperboard size={20} />
          </div>

          <div>
            <b>
              CINE<span>PREDICT</span>
            </b>
            <small>COLLABORATIVE FILM INTELLIGENCE</small>
          </div>
        </div>

        <div className="workspace-label">
          WORKSPACE <span>01</span>
        </div>

        <nav>
          {nav.map(item => (
            <button
              key={item.id}
              className={`nav-link ${page === item.id ? 'active' : ''}`}
              type="button"
              onClick={() => {
                setPage(item.id);
                setError('');
              }}
            >
              <item.icon size={17} />
              {item.label}
              {page === item.id && <i />}
            </button>
          ))}
        </nav>

        <div className="side-dataset">
          <div className="dataset-icon">
            <Layers3 size={17} />
          </div>

          <div>
            <b>MOVIELENS</b>
            <small>ML-LATEST-SMALL</small>
          </div>

          <span className="online-dot" />
        </div>

        <div className="sidebar-bottom">
          <div className="api-status">
            <span className={health ? 'online-dot' : 'offline-dot'} />
            <span>MODEL SERVICE</span>
            <b>{health ? 'ONLINE' : 'OFFLINE'}</b>
          </div>

          <p>
            Research prototype · Offline evaluation · No external tracking
          </p>

          <div className="side-version">
            CINEPREDICT LAB <span>v2.0</span>
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="breadcrumb">
            STUDIO <span>/</span> <b>{pageTitle[0]}</b>
          </div>

          <div className="top-actions">
            <span className="build-label">
              <i /> RESEARCH BUILD
            </span>
            <div className="avatar">CP</div>
          </div>
        </header>

        {/* DISCOVER PAGE */}
        {page === 'discover' && (
          <>
            <section className="hero">
              <div className="hero-content">
                <div className="eyebrow">
                  <span /> MOVIELENS / COLLABORATIVE FILTERING
                </div>

                <h1>
                  Stories find
                  <br />
                  their <em>people.</em>
                </h1>

                <p>
                  A recommendation engine that learns from the collective
                  taste of movie lovers. Discover films through the patterns
                  hidden in 100,836 ratings.
                </p>

                <div className="hero-cta">
                  <button
                    type="button"
                    onClick={() => setPage('new')}
                  >
                    Build your taste profile <ArrowUpRight size={15} />
                  </button>

                  <button
                    className="quiet-btn"
                    type="button"
                    onClick={() => setPage('research')}
                  >
                    Explore the research <ArrowRight size={14} />
                  </button>
                </div>

                <div className="hero-footnote">
                  <span>
                    <ShieldCheck size={13} /> TRAIN-ONLY SIMILARITY
                  </span>
                  <span>
                    <Activity size={13} /> TOP-10 RANKING
                  </span>
                  <span>
                    <Mic2 size={13} /> HUMAN TASTE, NOT METADATA
                  </span>
                </div>
              </div>

              <div className="hero-visual">
                <div className="hero-ring ring-one" />
                <div className="hero-ring ring-two" />

                <div className="hero-poster">
                  <Poster
                    movie={{ title: 'THE ART OF DISCOVERY (2026)' }}
                    index={2}
                  />
                  <span className="hero-poster-caption">
                    FIG. 01 <i /> TASTE IN MOTION
                  </span>
                </div>

                <div className="float-note">
                  <span className="float-icon">
                    <TrendingUp size={16} />
                  </span>

                  <div>
                    <b>0.245</b>
                    <small>BEST NDCG@10</small>
                  </div>
                </div>
              </div>
            </section>

            <section className="stats-strip">
              {[
                {
                  label: 'RATINGS IN THE MATRIX',
                  value: overview?.ratings?.toLocaleString() || '100,836',
                  icon: Activity
                },
                {
                  label: 'MOVIE LOVERS',
                  value: overview?.users?.toLocaleString() || '610',
                  icon: Users
                },
                {
                  label: 'RATED MOVIES',
                  value: overview?.movies_rated?.toLocaleString() || '9,724',
                  icon: Film
                },
                {
                  label: 'MATRIX SPARSITY',
                  value: overview
                    ? `${(overview.sparsity * 100).toFixed(1)}%`
                    : '98.3%',
                  icon: Layers3
                }
              ].map(stat => (
                <div className="stat" key={stat.label}>
                  <stat.icon size={16} />
                  <div>
                    <small>{stat.label}</small>
                    <b>{stat.value}</b>
                  </div>
                </div>
              ))}
            </section>

            <section className="feature-section">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">CHOOSE YOUR DISCOVERY PATH</div>
                  <h2>
                    Three ways to find <em>what's next.</em>
                  </h2>
                </div>
                <span className="section-index">01 — 03</span>
              </div>

              <div className="feature-grid">
                <button
                  className="feature-card"
                  type="button"
                  onClick={() => setPage('existing')}
                >
                  <div className="feature-icon">
                    <Users size={20} />
                  </div>
                  <span className="feature-no">01 / EXISTING USER</span>
                  <h3>Know the viewer.</h3>
                  <p>
                    Explore personalised top-10 recommendations for one of
                    the 610 MovieLens users.
                  </p>
                  <span className="feature-link">
                    Open user explorer <ArrowUpRight size={14} />
                  </span>
                </button>

                <button
                  className="feature-card"
                  type="button"
                  onClick={() => setPage('similar')}
                >
                  <div className="feature-icon">
                    <Layers3 size={20} />
                  </div>
                  <span className="feature-no">02 / ITEM SIMILARITY</span>
                  <h3>Follow a feeling.</h3>
                  <p>
                    Find movies connected by the rating behaviour of people
                    who watched both.
                  </p>
                  <span className="feature-link">
                    Find similar films <ArrowUpRight size={14} />
                  </span>
                </button>

                <button
                  className="feature-card"
                  type="button"
                  onClick={() => setPage('new')}
                >
                  <div className="feature-icon">
                    <Sparkles size={20} />
                  </div>
                  <span className="feature-no">03 / NEW USER</span>
                  <h3>Start with a few.</h3>
                  <p>
                    Rate a handful of films. The item-neighbourhood model
                    can recommend without retraining.
                  </p>
                  <span className="feature-link">
                    Create taste profile <ArrowUpRight size={14} />
                  </span>
                </button>
              </div>
            </section>

            <section className="research-callout">
              <div className="callout-art">
                <div className="callout-ring" />
                <Film size={32} />
              </div>

              <div>
                <div className="eyebrow">THE RESEARCH HIGHLIGHT</div>
                <h2>Ranking is not the same as predicting.</h2>
                <p>
                  In the paper's ablation, changing the ranking score moved
                  NDCG@10 from 0.022 to 0.245 — over 11× higher for the same
                  item-based model.
                </p>
              </div>

              <button
                type="button"
                onClick={() => setPage('research')}
              >
                Read the findings <ArrowUpRight size={15} />
              </button>
            </section>
          </>
        )}

        {/* EXISTING USER PAGE */}
        {page === 'existing' && (
          <section className="page-content">
            <PageHeading
              eyebrow={pageTitle[0]}
              title={pageTitle[1]}
              desc="Select a MovieLens user to see their strongest ratings and a ranked list of films they have not rated yet."
            />

            <div className="control-panel">
              <form className="inline-form" onSubmit={loadUser}>
                <label>
                  MOVIELENS USER ID
                  <input
                    type="number"
                    min="1"
                    max="610"
                    value={userId}
                    onChange={event => setUserId(event.target.value)}
                  />
                </label>

                <label>
                  RECOMMENDATIONS
                  <select
                    value={limit}
                    onChange={event => setLimit(+event.target.value)}
                  >
                    <option value="10">Top 10</option>
                    <option value="15">Top 15</option>
                    <option value="20">Top 20</option>
                  </select>
                </label>

                <label>
                  MINIMUM RATINGS
                  <select
                    value={minRatings}
                    onChange={event => setMinRatings(+event.target.value)}
                  >
                    <option value="0">All movies</option>
                    <option value="5">5+ ratings</option>
                    <option value="20">20+ ratings</option>
                  </select>
                </label>

                <button className="gold-btn" disabled={busy}>
                  {busy ? (
                    <LoaderCircle className="spin" size={16} />
                  ) : (
                    <Sparkles size={16} />
                  )}
                  Generate list
                </button>
              </form>
            </div>

            {userResult && (
              <>
                <div className="result-summary">
                  <div>
                    <span className="eyebrow">VIEWER PROFILE</span>
                    <h2>
                      MovieLens user <em>#{userResult.user_id}</em>
                    </h2>
                    <p>
                      {userResult.ratings_count} films rated ·
                      recommendations exclude titles already seen
                    </p>
                  </div>

                  <div className="profile-number">
                    <b>{userResult.ratings_count}</b>
                    <small>RATINGS FOUND</small>
                  </div>
                </div>

                <div className="subsection-head">
                  <h2>Highest-rated films</h2>
                  <span>VIEWER TASTE / TOP 8</span>
                </div>

                <div className="movie-grid compact-grid">
                  {userResult.top_rated.map((movie, index) => (
                    <MovieCard
                      key={movie.movie_id}
                      movie={{
                        ...movie,
                        average_rating: movie.estimated_rating
                      }}
                      index={index}
                      isSaved={isMovieSaved(movie)}
                      onToggleSave={toggleSavedMovie}
                    />
                  ))}
                </div>

                <div className="subsection-head">
                  <h2>Recommended for this viewer</h2>
                  <span>SIMILARITY-SUM RANKING / TOP {limit}</span>
                </div>

                <div className="movie-grid">
                  {userResult.recommendations.map((movie, index) => (
                    <MovieCard
                      key={movie.movie_id}
                      movie={movie}
                      index={index}
                      tag={`#${String(index + 1).padStart(2, '0')}`}
                      onSimilar={loadSimilar}
                      isSaved={isMovieSaved(movie)}
                      onToggleSave={toggleSavedMovie}
                    />
                  ))}
                </div>
              </>
            )}
          </section>
        )}

        {/* SIMILAR MOVIES PAGE */}
        {page === 'similar' && (
          <section className="page-content">
            <PageHeading
              eyebrow={pageTitle[0]}
              title={pageTitle[1]}
              desc="Item-based collaborative filtering compares residual rating patterns across people who rated both films. Pick a title to inspect its nearest neighbours."
            />

            <div className="search-panel">
              <div className="movie-search">
                <div className="search-box">
                  <Search size={17} />
                  <input
                    placeholder="Search the MovieLens catalogue…"
                    value={movieQuery}
                    onChange={event => {
                      setMovieQuery(event.target.value);
                      setSelectedMovie(null);
                      setSimilar(null);
                    }}
                  />
                </div>

                {movieResults.length > 0 && (
                  <div className="search-results">
                    {movieResults.map(movie => (
                      <button
                        key={movie.movie_id}
                        type="button"
                        onClick={() => {
                          setMovieQuery(movie.title);
                          setMovieResults([]);
                          loadSimilar(movie);
                        }}
                      >
                        <span>{movie.title}</span>
                        <small>{movie.genres}</small>
                        <ArrowUpRight size={14} />
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {similar && (
              <>
                <div className="selected-movie">
                  <Poster movie={similar.source} index={2} />

                  <div>
                    <span className="eyebrow">SELECTED FILM</span>
                    <h2>{similar.source.title}</h2>
                    <p>
                      {similar.source.genres?.replaceAll(' · ', ' / ')}
                    </p>

                    <div className="movie-meta">
                      <Rating value={similar.source.average_rating} />
                      <span>
                        {similar.source.rating_count} user ratings
                      </span>
                    </div>
                  </div>

                  <div className="selected-stat">
                    <b>{similar.similar.length}</b>
                    <small>NEIGHBOURS FOUND</small>
                  </div>
                </div>

                <div className="subsection-head">
                  <h2>Films with a similar audience signal</h2>
                  <span>RESIDUAL COSINE SIMILARITY</span>
                </div>

                <div className="movie-grid">
                  {similar.similar.map((movie, index) => (
                    <MovieCard
                      key={movie.movie_id}
                      movie={movie}
                      index={index}
                      tag={`SIM ${movie.similarity_score.toFixed(2)}`}
                      onSimilar={loadSimilar}
                      isSaved={isMovieSaved(movie)}
                      onToggleSave={toggleSavedMovie}
                    />
                  ))}
                </div>
              </>
            )}
          </section>
        )}

        {/* NEW USER / TASTE PROFILE PAGE */}
        {page === 'new' && (
          <section className="page-content">
            <PageHeading
              eyebrow={pageTitle[0]}
              title={pageTitle[1]}
              desc="Add a few films you've watched and rate them from 0.5 to 5. The model will use the same precomputed item similarities to build your list — no retraining required."
            />

            <div className="new-user-layout">
              <div className="control-panel add-panel">
                <div className="panel-title">
                  <span className="eyebrow">01 / BUILD YOUR PROFILE</span>
                  <h2>What have you watched?</h2>
                </div>

                {/* Updated movie search dropdown */}
                <div className="movie-search">
                  <div className="search-box">
                    <Search size={17} />
                    <input
                      placeholder="Search a movie to add…"
                      value={newQuery}
                      onChange={event => setNewQuery(event.target.value)}
                    />
                  </div>

                  {newResults.length > 0 && (
                    <div className="search-results">
                      {newResults.map(movie => (
                        <button
                          key={movie.movie_id}
                          type="button"
                          onClick={() => addRating(movie)}
                        >
                          <span>{movie.title}</span>
                          <small>{movie.genres}</small>
                          <span className="add-label">+ ADD</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <div className="rating-list">
                  {newRatings.map(movie => (
                    <div className="rating-row" key={movie.movie_id}>
                      <div>
                        <b>{movie.title}</b>
                        <small>YOUR RATING</small>
                      </div>

                      <select
                        value={movie.rating}
                        onChange={event =>
                          setNewRatings(old =>
                            old.map(item =>
                              item.movie_id === movie.movie_id
                                ? {
                                    ...item,
                                    rating: +event.target.value
                                  }
                                : item
                            )
                          )
                        }
                      >
                        {[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5].map(
                          rating => (
                            <option key={rating} value={rating}>
                              {rating.toFixed(1)} / 5
                            </option>
                          )
                        )}
                      </select>

                      <button
                        type="button"
                        onClick={() =>
                          setNewRatings(old =>
                            old.filter(
                              item => item.movie_id !== movie.movie_id
                            )
                          )
                        }
                      >
                        <X size={15} />
                      </button>
                    </div>
                  ))}
                </div>

                <div className="profile-foot">
                  <span>{newRatings.length} FILMS ADDED</span>

                  <button
                    className="gold-btn"
                    disabled={!newRatings.length || busy}
                    type="button"
                    onClick={getNewRecommendations}
                  >
                    {busy ? (
                      <LoaderCircle className="spin" size={16} />
                    ) : (
                      <Sparkles size={16} />
                    )}
                    Recommend for me
                  </button>
                </div>
              </div>

              <div className="profile-aside">
                <div className="aside-illustration">
                  <div className="big-orbit" />
                  <Clapperboard size={40} />
                </div>

                <h3>Your taste, translated.</h3>

                <p>
                  We compare your highly-rated films with the item
                  neighbourhoods learned from 100,836 historical ratings.
                </p>

                <div className="method-steps">
                  <span><i>01</i> Add movies you've seen</span>
                  <span><i>02</i> Rate them honestly</span>
                  <span><i>03</i> Discover unseen films</span>
                </div>
              </div>
            </div>

            {newRecommendations && (
              <>
                <div className="subsection-head">
                  <h2>Your recommendations</h2>
                  <span>
                    BASED ON {newRecommendations.ratings_count} RATINGS
                  </span>
                </div>

                <div className="movie-grid">
                  {newRecommendations.recommendations.map(
                    (movie, index) => (
                      <MovieCard
                        key={movie.movie_id}
                        movie={movie}
                        index={index}
                        tag={`#${String(index + 1).padStart(2, '0')}`}
                        onSimilar={loadSimilar}
                        isSaved={isMovieSaved(movie)}
                        onToggleSave={toggleSavedMovie}
                      />
                    )
                  )}
                </div>
              </>
            )}
          </section>
        )}

        {/* SAVED MOVIES PAGE */}
        {page === 'saved' && (
          <section className="page-content">
            <PageHeading
              eyebrow={pageTitle[0]}
              title={pageTitle[1]}
              desc="Your bookmarked films are stored in this browser, so they remain here when you refresh or reopen CinePredict."
            />

            <div className="subsection-head">
              <h2>Your watchlist</h2>
              <span>{savedMovies.length} SAVED {savedMovies.length === 1 ? 'FILM' : 'FILMS'}</span>
            </div>

            {savedMovies.length > 0 ? (
              <div className="movie-grid">
                {savedMovies.map((movie, index) => (
                  <MovieCard
                    key={movie.movie_id}
                    movie={movie}
                    index={index}
                    isSaved
                    onToggleSave={toggleSavedMovie}
                    onSimilar={loadSimilar}
                  />
                ))}
              </div>
            ) : (
              <div className="control-panel">
                <h2>No saved movies yet.</h2>
                <p>Use the bookmark icon on any movie card to add it to this collection.</p>
                <button
                  className="gold-btn"
                  type="button"
                  onClick={() => setPage('discover')}
                >
                  Discover movies <ArrowRight size={15} />
                </button>
              </div>
            )}
          </section>
        )}

        {/* RESEARCH PAGE */}
        {page === 'research' && (
          <section className="page-content">
            <PageHeading
              eyebrow={pageTitle[0]}
              title={pageTitle[1]}
              desc="A comparative study of neighbourhood and latent-factor methods on MovieLens ml-latest-small. The values below reproduce the accompanying paper's reported offline evaluation."
            />

            <div className="research-kpis">
              <div>
                <small>BEST RMSE</small>
                <b>0.854</b>
                <span>Item-based KNN</span>
              </div>

              <div>
                <small>BEST NDCG@10</small>
                <b>0.245</b>
                <span>Similarity-sum ranking</span>
              </div>

              <div>
                <small>HIT RATE@10</small>
                <b>70.6%</b>
                <span>Item-based KNN</span>
              </div>

              <div>
                <small>CATALOGUE COVERAGE</small>
                <b>5.2%</b>
                <span>Item-based KNN</span>
              </div>
            </div>

            <div className="research-block">
              <div className="subsection-head">
                <h2>Model comparison</h2>
                <span>TABLE II / MEAN ± STD OVER THREE SPLITS</span>
              </div>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>MODEL</th>
                      <th>RMSE ↓</th>
                      <th>MAE ↓</th>
                      <th>P@10 ↑</th>
                      <th>R@10 ↑</th>
                      <th>NDCG@10 ↑</th>
                      <th>HR@10 ↑</th>
                    </tr>
                  </thead>

                  <tbody>
                    {evaluation?.models?.map(model => (
                      <tr
                        key={model.name}
                        className={model.best ? 'best-row' : ''}
                      >
                        <td>
                          {model.name}
                          {model.best && (
                            <span className="best-tag">BEST</span>
                          )}
                        </td>
                        <td>{model.rmse}</td>
                        <td>{model.mae}</td>
                        <td>{model.precision}</td>
                        <td>{model.recall}</td>
                        <td>{model.ndcg}</td>
                        <td>{model.hit_rate}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="research-block">
              <div className="subsection-head">
                <h2>Why the ranking score matters</h2>
                <span>TABLE III / ABLATION</span>
              </div>

              <div className="ablation-grid">
                {evaluation?.ablation?.map((item, index) => (
                  <div
                    className={`ablation-card ${
                      index === 0 ? 'highlight' : ''
                    }`}
                    key={item.variant}
                  >
                    <small>{item.variant}</small>

                    <div>
                      <b>{item.ndcg}</b>
                      <span>NDCG@10</span>
                    </div>

                    <div className="mini-bar">
                      <i
                        style={{
                          width: `${(Number(item.ndcg) / 0.245) * 100}%`
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="methodology-grid">
              <div className="method-card">
                <span className="eyebrow">DATASET</span>
                <h3>MovieLens ml-latest-small</h3>
                <p>
                  100,836 ratings from 610 users across 9,724 rated movies.
                  The user–movie matrix is 98.3% sparse.
                </p>
              </div>

              <div className="method-card">
                <span className="eyebrow">VALIDATION</span>
                <h3>Leakage-aware tuning</h3>
                <p>
                  20% per-user holdout for test; a further 10% of training
                  for validation. Seeds 42, 1 and 2.
                </p>
              </div>

              <div className="method-card">
                <span className="eyebrow">ITEM SIMILARITY</span>
                <h3>Residual cosine + shrinkage</h3>
                <p>
                  Subtract user and item biases before cosine similarity,
                  then shrink low-overlap similarities toward zero.
                </p>
              </div>
            </div>

            <div className="limitations">
              <ShieldCheck size={18} />

              <div>
                <b>Interpret the results with care</b>
                <p>
                  One small development snapshot, random per-user holdout,
                  offline metrics only, no user study, and cold-start
                  limitations. The paper notes that unobserved movies are
                  treated as non-relevant and that SVD was only tested with
                  predicted-rating ranking.
                </p>
              </div>
            </div>
          </section>
        )}

        {error && (
          <div className="error-toast">
            <X size={15} onClick={() => setError('')} />
            {error}
          </div>
        )}

        <footer className="footer">
          <span>© 2026 CINEPREDICT RESEARCH LAB</span>
          <span>COLLABORATIVE FILTERING · MOVIELENS</span>
          <span>
            BUILT ON SHARED TASTE <span className="gold">✳</span>
          </span>
        </footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
