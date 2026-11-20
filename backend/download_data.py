"""Download MovieLens ml-latest-small from GroupLens into data/ (run once, needs internet).
Data (c) GroupLens Research, University of Minnesota. Cite: Harper & Konstan, ACM TiiS 5(4), 2015, doi:10.1145/2827872.
Do not redistribute the data; see the README that ships inside the zip for the licence terms."""
import io, os, urllib.request, zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
os.makedirs("data", exist_ok=True)
with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read())) as z:
    for name in ("ratings.csv", "movies.csv"):
        with open(f"data/{name}", "wb") as f:
            f.write(z.read(f"ml-latest-small/{name}"))
print("Saved data/ratings.csv and data/movies.csv")
