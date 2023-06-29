
from typing import Union, List
from sentence_transformers import CrossEncoder
from collections import OrderedDict
import pandas as pd
from lib.searchconsole import load_gsc_account_data, load_available_gsc_accounts
from lib.nlp import clean_gsc_dataframe, get_ngram_frequency, merge_ngrams, filter_knee, get_structure
from lib.api import get_openai_response_chat, get_palm_response
from lib.prompts import PROMPT_TEMPLATE_TAXONOMY, PROMPT_TEMPLATE_TAXONOMY_LLM_DESCRIPTIONS
from lib.utils import create_tuples
from lib.clustering import ClusterTopics
from loguru import logger
import settings



def get_data(data: Union[str,pd.DataFrame],
             text_column: str = None,
             search_volume_column: str = None,
             days: int = 30, 
             brand: str = None, 
             limit_queries: int = 5) -> pd.DataFrame:
    """Get data from Google Search Console or a pandas dataframe."""

    df = pd.DataFrame()

    if isinstance(data, str) and ("sc-domain:" in data or "https://" in data):
        df = load_gsc_account_data(data, days)

        if df is None:
            df_accounts = load_available_gsc_accounts()
            accounts = df_accounts[df_accounts["property"].str.contains(data)]["property"].tolist()
            if len(accounts) == 0:
                raise ValueError("No GSC account found.")
            elif len(accounts) > 1:
                logger.warning(f"Multiple accounts found. {', '.join(accounts)}.")
                account = input("Which account would you like to use? ")
                df = load_gsc_account_data(account, days)
            else:
                logger.info(f"GSC account found. Using: {accounts[0]}")
                df = load_gsc_account_data(accounts[0], days)

        df = clean_gsc_dataframe(df, brand, limit_queries)



    elif isinstance(data, pd.DataFrame) or isinstance(data, str) and (".csv" in data):

        if isinstance(data, str) and (".csv" in data):
            df = pd.read_csv(data)
        else:
            df = data

        if text_column is None:
            text_column = input("What is the name of the column with the queries? ")
        
        if search_volume_column is None:
            search_volume_column = input("What is the name of the column with the search volume? ")
        

        # Rename columns
        df = df.rename(columns={text_column: "query", search_volume_column: "search_volume"})

        # Remove other columns
        df = df[["query", "search_volume"]]

        df['original_query'] = df['query']

        # Remove non-english characters from query using regex: [^a-zA-Z0-9\s]
        df["query"] = df["query"].str.replace(r'[^a-zA-Z0-9\s]', '')

        # Trim whitespace from query
        df["query"] = df["query"].str.strip()

        # Remove rows where query is empty
        df = df[df["query"] != '']

        # Convert search volume to int
        df["search_volume"] = df["search_volume"].fillna(0).astype(int)

        # Remove rows where search volume is empty or na
        df = df[df["search_volume"].notna()]

    else:
        raise ValueError("Data must be a GSC Property, CSV Filename, or pandas dataframe.")


    return df


def score_and_filter_df(df: pd.DataFrame,
                        ngram_range: tuple = (1, 6),
                        filter_knee: bool = True,
                        S: int = 100,
                        min_df: int = 2,) -> pd.DataFrame:
    """Score and filter dataframe."""

    df_ngram = get_ngram_frequency(df['query'].tolist(), ngram_range=ngram_range, min_df=min_df)
    logger.info(f"Got ngram frequency. Dataframe shape: {df_ngram.shape}")

    df_ngram = merge_ngrams(df_ngram)
    logger.info(f"Merged Ngrams. Dataframe shape: {df_ngram.shape}")

    df_ngram = df_ngram.rename(columns={"feature": "query"})

    # Merge with original dataframe
    df_ngram = df_ngram.merge(df, on="query", how="left")

    # Drop duplicates
    df_ngram = df_ngram.drop_duplicates(subset=["query"])

    # drop if any column is na
    df_ngram = df_ngram.dropna()

    # Normalize the columns: search_volume,  frequency,  merged_frequency,  ngram_size
    df_ngram["search_volume"] = df_ngram["search_volume"] / df_ngram["search_volume"].max()
    df_ngram["frequency"] = df_ngram["frequency"] / df_ngram["frequency"].max()
    df_ngram["merged_frequency"] = df_ngram["merged_frequency"] / df_ngram["merged_frequency"].max()


    #Updata score column to be the average of the normalized columns
    df_ngram["score"] = df_ngram[["search_volume", "frequency", "merged_frequency"]].mean(axis=1)

    # Sort by score
    df_ngram = df_ngram.sort_values(by=["score"], ascending=False)

    df_ngram = df_ngram.reset_index(drop=True)

    if filter_knee:
        df_ngram = filter_knee(df_ngram, col_name="score", S=S)
        logger.info(f"Filtered Knee (sensitivity={S}). Dataframe shape: {df_ngram.shape}")

    return df_ngram





def create_taxonomy(data: Union[str, pd.DataFrame],
                    text_column: str = None,
                    search_volume_column: str = None,
                    platform: str = "palm", # "palm" or "openai"
                    use_llm_descriptions: bool = False,
                    days: int = 30,
                    ngram_range: tuple = (1, 6),
                    min_df: int = 2,
                    S: int = 500,
                    brand: str = None,
                    limit_queries: int = 5,
                    debug_responses: bool = False):
    """Kickoff function to create taxonomy from GSC data."""

    # Get data
    df = get_data(data, text_column, search_volume_column, days, brand, limit_queries)
    logger.info(f"Got Data. Dataframe shape: {df.shape}")


    if use_llm_descriptions:
        logger.info("Using LLM Descriptions.")
        # Get ngram frequency
        df_ngram = score_and_filter_df(df, ngram_range=ngram_range, filter_knee=False, min_df=min_df)
        logger.info(f"Got ngram frequency. Dataframe shape: {df_ngram.shape}")
        queries = list(set(df_ngram["query"].tolist()))

        cluster_model = ClusterTopics(
                                        embedding_model = platform,
                                        min_cluster_size = 5,
                                        min_samples = 3,
                                        reduction_dims  = 5,
                                        cluster_model = "agglomerative",
                                        cluster_description_model = platform
                                    )
        
        labels, text_labels = cluster_model.fit(queries)
        label_lookup = {query: label for query, label in zip(queries, text_labels)}

        def lookup_label(query):
            if query not in label_lookup:
                return "<UNK>"
            return label_lookup[query]

        df_ngram["description"] = df_ngram["query"].map(lookup_label)

        samples = list(set(text_labels))
        logger.info(f"Got samples. Number of samples: {len(samples)}")
        prompt = PROMPT_TEMPLATE_TAXONOMY_LLM_DESCRIPTIONS.format(samples=samples, brand=brand)

    else:

        logger.info("Using Elbow to define top ngram queries.")
        df_ngram = score_and_filter_df(df, ngram_range=ngram_range, min_df=min_df, S=S)
        logger.info(f"Got ngram frequency. Dataframe shape: {df_ngram.shape}")
        samples = list(set(df_ngram["query"].tolist()))[:900]
        logger.info(f"Got samples. Number of samples: {len(samples)}")
        prompt = PROMPT_TEMPLATE_TAXONOMY.format(samples=samples, brand=brand)


    # Get response    
    if platform == "palm":
        logger.info("Using Palm API.")
        response = get_palm_response(prompt)
    elif platform == "openai":
        logger.info("Using OpenAI API.")
        response = get_openai_response_chat(prompt)
    else:
        raise ValueError("Platform must be 'palm' or 'openai'.")
    
    if not response:
        logger.error("No response from API.")
        return None
    
    if debug_responses:
        logger.info(response)
    
    # Get structure
    logger.info("Getting structure.")
    structure = get_structure(response)

    logger.info("Done.")

    return structure, df, samples



def add_categories(taxonomy:List[str], df: pd.DataFrame, brand: Union[str, None] = None) -> pd.DataFrame:
    """Add categories to dataframe."""

    logger.info("Finding Categories")


    queries = list(set(df['original_query'].tolist()))
    if brand:
        taxonomies = [brand] + taxonomy.copy()
    else:
        taxonomies = taxonomy.copy()


    # Use ordered dict to keep only unique terms of t.split(" > ") in taxonomies.
    categories = [" ".join(OrderedDict.fromkeys(t.split(" > ")).keys()) for t in taxonomies]
    
    model = CrossEncoder(settings.CROSSENCODER_MODEL_NAME, max_length=128)

    compare_pairs = create_tuples(categories, queries)

    logger.info(f"Comparing {len(compare_pairs)} items with cross-encoding.")

    scores = model.predict(compare_pairs, batch_size=512)

    df_category = pd.DataFrame({"scores": scores,
                                "categories": [s[0] for s in compare_pairs],
                                "queries": [s[1] for s in compare_pairs]})

    df_category.sort_values(by="scores", ascending=False, inplace=True)

    # This assigns the most similar category to each query
    df_category = df_category.groupby(["queries"], as_index=False).agg({'categories': 'first', 'scores': 'first'})

    df_category.columns = ['original_query', 'taxonomy_category', 'similiary_score']

    df_out = df.merge(df_category, on="original_query", how="left")

    # Lookup and add back original taxonomy
    df_out['taxonomy'] = df_out['taxonomy_category'].map(lambda x: taxonomies[categories.index(x)])

    return df_out


