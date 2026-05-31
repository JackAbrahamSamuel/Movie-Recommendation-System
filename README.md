# movie-recommender

My first real ML project. I wanted to learn how Netflix decides what to show you next, so I built a version of it myself using nothing but Python and some math. The idea is simple — you type a movie name and it finds similar ones based on genre, director, cast, and plot keywords. There's also fuzzy search so it doesn't break if you typo the title.

---

## ML Concepts

**TF-IDF** — instead of just counting words, it weighs them by how rare they are across all 45k movies. So "Christopher Nolan" means more than "Action" because half the dataset is action movies.

**Cosine similarity** — measures the angle between two movie vectors, not the distance. Two movies can have wildly different description lengths and still score as nearly identical if they point in the same direction.

**Content-based filtering** — recommendations come from the movie's own attributes, not from what other users watched. Means it works from day one with zero user data.

**Cold-start problem** — what happens when you have no user history to learn from. Content-based filtering sidesteps it entirely, which is why I went with this approach over collaborative filtering.

---

## Tech stack

Python, pandas, numpy, scikit-learn

---

## Dataset

TMDB 5000 Movie Dataset (Kaggle) — download the two CSVs and drop them in the project folder

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

---

## Setup

```bash
pip install -r requirements.txt
python recommender_tmdb.py
```

---

## How it works

```
  search: inception

  -> Inception

  Title                              Genres                  Rtng  Year
  --------------------------------------------------------------------
  Interstellar                       Drama / Science Fictio   8.6  2014
  The Dark Knight                    Action / Crime / Thril   9.0  2008
  The Prestige                       Drama / Mystery / Thri   8.5  2006
  Memento                            Mystery / Thriller         8.4  2000
  Batman Begins                      Action / Crime / Drama   7.7  2005
```
