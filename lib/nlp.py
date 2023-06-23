from typing import List
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm
from nltk.util import ngrams
from kneed import KneeLocator

import settings


model = CrossEncoder(settings.CROSSENCODER_MODEL_NAME, max_length=128)


def get_structure(text: str) -> List[str]:
    """Get the structure of a text document."""

    path = []
    result = []

    # Calculate levels of indentation
    indents = []
    for line in text.split("\n"):
        if line.strip() == "":
            continue
        depth = len(line) - len(line.lstrip())
        indents.append(depth)
    
    # Convert indents to levels
    indents = sorted(list(set(indents)))
    levels = {indents[i]: i for i in range(len(indents))}

    for line in text.split("\n"):
        if line.strip() == "":
            continue
        depth = len(line) - len(line.lstrip())
        name = line.strip().replace("-", "").strip()
        level = levels[depth]

        while level < len(path):
            path.pop()
        path.append(name)
        if len(path) > 1:
            result.append(" > ".join(path))
    return result



def filter_knee(df: pd.DataFrame, S: int = 100) -> pd.DataFrame:   
    kneedle = KneeLocator(range(1, len(df) + 1), df["score"], curve="convex", direction="decreasing", S=S)
    df_knee = df.iloc[:kneedle.knee]
    return df_knee


def merge_ngrams(df: pd.DataFrame):
    """Merge lower order ngrams into higher order ngrams based on frequency score."""
    ngram_dict = {}

    # Get the range of ngram sizes
    ngram_range = (df["feature"].str.split().str.len().min(), df["feature"].str.split().str.len().max())

    # Build a dict of ngrams and their scores for each ngram size
    for ngram_size in range(ngram_range[0], ngram_range[1] + 1):
        ngram_mask = df["feature"].str.split().str.len() == ngram_size
        ngrams_df = df[ngram_mask]
        ngram_dict[ngram_size] = {row.feature: row.frequency for row in ngrams_df.itertuples()}

    # Loop over the ngram sizes from smallest to largest
    for ngram_size in range(ngram_range[0], ngram_range[1]):
        merge_candidates = ngram_dict[ngram_size]
        merge_into_candidates = ngram_dict[ngram_size + 1]

        # Loop over the ngrams of the current size
        for merge_candidate, merge_candidate_score in merge_candidates.items():

            # Get only the merge_into_candidates with score less than or equal to merge_candidate_score
            merge_into_candidates_reduced = {k: v for k, v in merge_into_candidates.items() if v <= merge_candidate_score and all(term in k.split() for term in merge_candidate.split())}

            # Loop over the larger ngrams
            for merge_into_candidate, merge_into_candidate_score in merge_into_candidates_reduced.items():

                # Create ngrams of the merge_into_candidate the same size as the merge_candidate
                merge_into_candidate_ngrams = [" ".join(n) for n in ngrams(merge_into_candidate.split(), ngram_size)]

                if merge_candidate in merge_into_candidate_ngrams:
                    # Reduce the score of the smaller ngram by the score of the larger ngram
                    merge_candidates[merge_candidate] -= merge_into_candidate_score

                    # If the score of the smaller ngram becomes negative, set it to 0
                    if merge_candidates[merge_candidate] < 0:
                        merge_candidates[merge_candidate] = 0

        # Remove any ngrams with a score of 0
        merge_candidates = {k: v for k, v in merge_candidates.items() if v != 0}

        # Update the dictionary with the new scores
        ngram_dict[ngram_size] = merge_candidates

    # Convert the dictionary back to a DataFrame
    df_out = pd.DataFrame([(k, sub_k, sub_v) for k, v in ngram_dict.items() for sub_k, sub_v in v.items()], columns=['ngram_size', 'feature', 'merged_frequency'])

    df_out = df_out.merge(df, on="feature", how="left")

    # Calculate the score for the merged ngrams
    df_out["score"] = df_out["merged_frequency"] * df_out["ngram_size"] * df_out["frequency"]

    # Normalize scores
    df_out["score"] = df_out["score"] / df_out["score"].max()

    # sort by score and reset index
    df_out = df_out.sort_values(by=["score"], ascending=False).reset_index(drop=True)

    return df_out


def get_ngram_frequency(texts: List[str], ngram_range: tuple = (1, 6), min_df: int = 2) -> pd.DataFrame:
    """Get ngram frequency from a dataframe."""

    # Find counts for each query
    cv = CountVectorizer(stop_words="english", ngram_range=ngram_range, min_df=min_df)
    cv.fit(texts)
    cv_matrix = cv.transform(texts)

    # Sum scores by feature name
    feature_names = cv.get_feature_names_out()
    scores = cv_matrix.sum(axis=0).tolist()[0]

    # Create a dataframe of feature names and frequency
    df_cv = pd.DataFrame(list(zip(feature_names, scores)), columns=["feature", "frequency"])

    # Sort by frequency
    df_cv = df_cv.sort_values(by=["frequency"], ascending=False).reset_index(drop=True)

    return df_cv



def clean_gsc_dataframe(df: pd.DataFrame, brand: str = None, limit_queries: int = 5) -> pd.DataFrame:
    """Clean up the GSC dataframe."""

    df['query'] = df['query'].str.lower()

    # Remove non-english characters from query
    df['query'] = df['query'].str.replace(r'[^a-zA-Z0-9\s]', '')

    # Trim whitespace from query
    df['query'] = df['query'].str.strip()

    # Remove rows where query is empty
    df = df[df['query'] != '']

    df['original_query'] = df['query']

    # Rename impressions to search_volume
    df = df.rename(columns={"impressions": "search_volume"})

    if brand:
        # Split brand into terms
        brand_terms = brand.lower().split(' ')
        df["query"] = df["query"].apply(lambda x: ' '.join([word for word in x.split(' ') if word.lower() not in (brand_terms)]))


    # Sort by clicks and impressions descending
    df = df.sort_values(by=['clicks', 'search_volume'], ascending=False)

    # Keep only the first 5 queries for each page. This is to avoid pages with a lot of queries from dominating the data
    if limit_queries:
        df = df.groupby("page").head(limit_queries)

    return df





    





