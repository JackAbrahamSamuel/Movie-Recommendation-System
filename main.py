import pandas as pd
import numpy as np
import ast, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(movies_path="tmdb_5000_movies.csv", credits_path="tmdb_5000_credits.csv"):

    print("loading data...")

    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        print("cant find the csv files!")

    movies = pd.read_csv(movies_path)
    creds  = pd.read_csv(credits_path)

    if "movie_id" in creds.columns:
        creds.rename(columns={"movie_id": "id"}, inplace=True)

    merged = movies.merge(creds, on="id")
    merged = merged[["id","title_x","genres","keywords","overview","cast","crew","vote_average","release_date"]]
    merged.rename(columns={"title_x": "title"}, inplace=True)

    print(f"loaded {len(merged):,} movies")
    return merged


def parse_col(txt):
    try:
        return ast.literal_eval(txt)
    except:
        return []

def get_names(lst, n=None):
    out = [x["name"] for x in lst if "name" in x]
    return out[:n] if n else out

def get_director(crew):
    for person in crew:
        if person.get("job") == "Director":
            return person.get("name","")
    return ""


def make_soup(row):
    g = " ".join(row["genres_clean"])
    kw = " ".join(row["kw_clean"])
    # join cast names with underscore so "Tom Hanks" stays as one token
    c = " ".join([name.replace(" ","_") for name in row["cast_clean"]])
    d = row["director"].replace(" ","_")
    ov = row["overview"] if isinstance(row["overview"], str) else ""
    return f"{g} {kw} {c} {d} {d} {ov}"


def preprocess(df):
    print("preprocessing...")
    data = df.copy()

    data["genres_clean"]  = data["genres"].apply(parse_col).apply(get_names)
    data["kw_clean"]      = data["keywords"].apply(parse_col).apply(lambda x: get_names(x, 5))
    data["cast_clean"]    = data["cast"].apply(parse_col).apply(lambda x: get_names(x, 3))
    data["director"]      = data["crew"].apply(parse_col).apply(get_director)
    data["soup"]          = data.apply(make_soup, axis=1)

    data = data[data["soup"].str.strip() != ""].reset_index(drop=True)
    data["year"] = pd.to_datetime(data["release_date"], errors="coerce").dt.year

    print(f"done, {len(data):,} movies remaining")
    return data


def build_tfidf(data):
    print("building tfidf matrix, might take a few seconds...")

    vec = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1,2))
    mat = vec.fit_transform(data["soup"])

    print(f"matrix: {mat.shape}")
    return mat


class TitleSearch:

    def __init__(self, titles):
        self.titles = titles
        self.v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))
        self.m = self.v.fit_transform(titles.str.lower())

    def find(self, q, k=5):
        qv     = self.v.transform([q.lower()])
        scores = cosine_similarity(qv, self.m).flatten()
        idxs   = scores.argsort()[::-1][:k]
        return [(i, self.titles.iloc[i], round(scores[i],3)) for i in idxs if scores[i] > 0]


def recommend(idx, data, mat, n=10):
    vec = mat[idx]
    sims = cosine_similarity(vec, mat).flatten()

    ranked = sims.argsort()[::-1]
    ranked = [i for i in ranked if i != idx][:n]

    res = data.iloc[ranked][["title","genres_clean","vote_average","year","director"]].copy()
    res["similarity"] = [round(sims[i], 3) for i in ranked]
    res["genres"]     = res["genres_clean"].apply(lambda g: " / ".join(g[:3]) if g else "?")
    res["year"]       = res["year"].fillna(0).astype(int).replace(0,"N/A")

    return res[["title","genres","vote_average","year","director","similarity"]]


def run(data, mat, searcher):

    print("\n" + "-"*50)
    print("  movie recommender | 45k movies")
    print("  type a movie name, or 'q' to quit")
    print("-"*50)

    while True:
        q = input("\n  search: ").strip()

        if q.lower() in ("q","quit","exit"):
            print("bye!")
            break

        if not q:
          continue

        hits = searcher.find(q, k=5)

        if not hits:
            print("  nothing found, try again")
            continue

        top_idx, top_title, top_score = hits[0]

        if top_score > 0.85:
            chosen = top_idx
            print(f"\n  -> {top_title}")
        else:
            print("\n  did you mean:")
            for i,(idx,title,score) in enumerate(hits, 1):
                yr = data.iloc[idx]["year"]
                yrstr = f"({int(yr)})" if pd.notna(yr) and yr != 0 else ""
                print(f"    {i}. {title} {yrstr}")

            pick = input("\n  pick a number (enter = first): ").strip()
            try:
                chosen = hits[int(pick)-1][0]
                print(f"  using: {hits[int(pick)-1][1]}")
            except:
                chosen = hits[0][0]
                print(f"  using: {hits[0][1]}")

        recs = recommend(chosen, data, mat, n=10)

        print(f"\n  {'Title':<35} {'Genres':<22} {'Rtng':>4}  Year")
        print("  " + "-"*68)
        for _,row in recs.iterrows():
            t = str(row["title"])[:34]
            g = str(row["genres"])[:21]
            r = f"{row['vote_average']:.1f}" if row["vote_average"] else " N/A"
            y = str(row["year"])
            print(f"  {t:<35} {g:<22} {r:>4}  {y}")


def main():

    data = load_data()
    data = preprocess(data)
    mat  = build_tfidf(data)

    print("indexing titles...")
    searcher = TitleSearch(data["title"])
    print("ready\n")

    # quick sanity check
    test = searcher.find("The Dark Knight", k=1)
    if test:
        print("sample recs for 'The Dark Knight':")
        print(recommend(test[0][0], data, mat, n=5).to_string(index=False))

    run(data, mat, searcher)


if __name__ == "__main__":
    main()