import requests
from tqdm import tqdm
from scipy import stats
from tokenizer_utils import get_tokenizer
from constants import AUTHORS, CLEANED_DATA_DIR, DATA_DIR
import pandas as pd


tokenizer = get_tokenizer("gpt2")
tokenizer.model_max_length = 1e8

df = {}
for author in tqdm(AUTHORS):
    book_paths = list(CLEANED_DATA_DIR.glob(f"{author}/*.txt"))
    for book_path in book_paths:
        book_id = book_path.stem

        # Handle non-US Gutenberg books
        stem_to_name = {
            "gutenberg_net_au_fsf_PAT-HOBBY": "the pat hobby stories",
            "gutenberg_net_au_ebooks03_0301261": "tender is the night",
        }
        if book_id in stem_to_name:
            title = stem_to_name[book_id]

        # Handle US Gutenberg books
        else:
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            text = requests.get(url)
            lines = text.text.splitlines()
            lines = [line.strip().lower() for line in lines]
            found = False
            for line in lines:
                if "the project gutenberg ebook of" in line:
                    title = line.split("the project gutenberg ebook of ")[1]
                    found = True
                    break
            if not found:
                raise Exception(f"Book title not found for {book_id}")

        # Get the number of tokens in the cleaned book
        with open(book_path, "r", encoding="utf-8") as f:
            text = f.read()
            tokens = tokenizer.encode(text)
            df[book_id] = {
                "author": author,
                "title": title,
                "n_tokens": len(tokens),
            }

# Save a CSV file with the number of tokens per book
df = pd.DataFrame.from_dict(df, orient="index")
df.to_csv(DATA_DIR / "book_stats.csv", index=False)

# Save another CSV file with the mean, std, count, and sum of tokens per author
df = df.groupby("author").agg({"n_tokens": ["mean", "std", "count", "sum"]})
df.columns = ["mean", "std", "count", "sum"]
df = df.reset_index()
df.to_csv(DATA_DIR / "book_stats_by_author.csv", index=False)
