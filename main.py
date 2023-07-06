
from typing import Union, List
from sentence_transformers import CrossEncoder
from collections import OrderedDict
import re
import pandas as pd
from lib.searchconsole import load_gsc_account_data, load_available_gsc_accounts
from lib.nlp import clean_gsc_dataframe, clean_provided_dataframe, get_ngram_frequency, merge_ngrams, filter_knee, get_structure
from lib.api import get_openai_response_chat, get_palm_response
from lib.prompts import PROMPT_TEMPLATE_TAXONOMY, PROMPT_TEMPLATE_TAXONOMY_LLM_DESCRIPTIONS, PROMPT_TEMPLATE_TAXONOMY_REVIEW
from lib.utils import create_tuples
from lib.clustering import ClusterTopics
from loguru import logger
import settings



def get_data(data: Union[str,pd.DataFrame],
             text_column: str = None,
             search_volume_column: str = None,
             days: int = 30, 
             brand_terms: Union[List[str], None] = None, 
             limit_queries: int = 5) -> pd.DataFrame:
    """Get data from Google Search Console or a pandas dataframe."""

    df = pd.DataFrame()
    df_original = pd.DataFrame()

    if isinstance(data, str) and ("sc-domain:" in data or "https://" in data):
        df = load_gsc_account_data(data, days)

        if df is None:
            df_accounts = load_available_gsc_accounts()
            accounts = df_accounts[df_accounts["property"].str.contains(data)]["property"].tolist()
            if len(accounts) == 0:
                raise AttributeError(f"No GSC account found for: {data}")
            elif len(accounts) > 1:
                logger.warning(f"Multiple accounts found. {', '.join(accounts)}.")
                account = input("Which account would you like to use? ")
                df = load_gsc_account_data(account, days)
            else:
                logger.info(f"GSC account found. Using: {accounts[0]}")
                df = load_gsc_account_data(accounts[0], days)

        df = clean_gsc_dataframe(df, brand_terms, limit_queries)



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

        # Save original dataframe
        df_original = df.copy()

        # Clean
        df = clean_provided_dataframe(df, brand_terms)

    else:
        raise ValueError("Data must be a GSC Property, CSV Filename, or pandas dataframe.")


    return df, df_original


def score_and_filter_df(df: pd.DataFrame,
                        ngram_range: tuple = (1, 6),
                        min_df: int = 2,) -> pd.DataFrame:
    """Score and filter dataframe."""

    df_ngram = get_ngram_frequency(df['query'].tolist(), ngram_range=ngram_range, min_df=min_df)
    logger.info(f"Got ngram frequency. Dataframe shape: {df_ngram.shape}")

    df_ngram = merge_ngrams(df_ngram)
    logger.info(f"Merged Ngrams. Dataframe shape: {df_ngram.shape}")

    df_ngram = df_ngram.rename(columns={"feature": "query"})

    # Sum search_volume column for df rows where query matches all terms in df_ngram["query"]. Do not use .str.contains() because may match parts of words.
    def match_all_terms(text, longer_text):
        t = text.lower().split(" ")
        lt = longer_text.lower().split(" ")
        return all([x in lt for x in t])
    
    df_ngram["search_volume"] = df_ngram["query"].apply(lambda x: df[df["query"].apply(lambda y: match_all_terms(x, y))]["search_volume"].sum())

    # Normalize the columns: search_volume
    df_ngram["search_volume"] = df_ngram["search_volume"] / df_ngram["search_volume"].max()
    #Updata score column to be the average of the normalized column
    df_ngram["score"] = df_ngram[["search_volume", "score"]].mean(axis=1)

    # Sort by score
    df_ngram = df_ngram.sort_values(by=["score"], ascending=False)

    df_ngram = df_ngram.reset_index(drop=True)

    if len(df_ngram) <= settings.MAX_SAMPLES:
        logger.info(f"Final score and filter length: {len(df_ngram)}")
        print(df_ngram.head())
        return df_ngram
    

    df_knee = None
    S = settings.MAX_SAMPLES

    # Filter by knee
    while df_knee is None or len(df_knee) > settings.MAX_SAMPLES:
        df_knee = filter_knee(df_ngram.copy(), col_name="score", S=S)
        S -= 100
    
    logger.info(f"Filtered Knee (sensitivity={S}). Dataframe shape: {df_knee.shape}")
    print(df_knee.head())

    return df_knee





def create_taxonomy(data: Union[str, pd.DataFrame],
                    text_column: str = None,
                    search_volume_column: str = None,
                    taxonomy_model: str = "palm", # "palm" or "openai"
                    use_clustering: bool = False,
                    use_llm_cluster_descriptions: bool = False,
                    cluster_embeddings_model: Union[str, None] = None,
                    min_cluster_size: int = 5,
                    min_samples: int = 2,
                    days: int = 30,
                    ngram_range: tuple = (1, 6),
                    min_df: int = 2,
                    brand_terms: List[str] = None,
                    limit_queries: int = 5,
                    debug_responses: bool = False):
    """Kickoff function to create taxonomy from GSC data.

    Args:
        data (Union[str, pd.DataFrame]): GSC Property, CSV Filename, or pandas dataframe.
        text_column (str, optional): Name of the column with the queries. Defaults to None.
        search_volume_column (str, optional): Name of the column with the search volume. Defaults to None.
        taxonomy_model (str, optional): Name of the taxonomy model. Defaults to "palm".
        use_clustering (bool, optional): Whether to use clustering. Defaults to False.
        use_llm_cluster_descriptions (bool, optional): Whether to use LLM descriptions for clustering. Defaults to False.
        cluster_embeddings_model (Union[str, None], optional): Name of the cluster embeddings model. Defaults to None.
        min_cluster_size (int, optional): Minimum cluster size to use for clustering. Defaults to 5.
        min_samples (int, optional): Minimum samples to use for clustering. Defaults to 2.
        days (int, optional): Number of days to get data from. Defaults to 30.
        ngram_range (tuple, optional): Ngram range to use for scoring. Defaults to (1, 6).
        min_df (int, optional): Minimum document frequency to use for scoring. Defaults to 2.
        brand_terms (List[str], optional): List of brand terms to remove from queries. Defaults to None.
        limit_queries (int, optional): Number of queries to use for clustering. Defaults to 5.
        debug_responses (bool, optional): Whether to print debug responses. Defaults to False.

    Returns:
        structure, df, samples
        Tuple[List[str], pd.DataFrame, List[str]]: Taxonomy list, original dataframe, and sample queries.
    """

    # Get data
    df, df_original = get_data(data, text_column, search_volume_column, days, brand_terms, limit_queries)
    logger.info(f"Got Data. Dataframe shape: {df.shape}")


    if use_clustering:

        logger.info("Using LLM Descriptions.")
        # Get ngram frequency
        logger.info(f"Dataframe shape: {df.shape}")
        queries = list(set(df["query"].tolist()))

        cluster_model = ClusterTopics(
                                        embedding_model = cluster_embeddings_model,
                                        min_cluster_size = min_cluster_size,
                                        min_samples = min_samples,
                                        reduction_dims  = 5,
                                        cluster_model = "hdbscan",
                                        use_llm_descriptions = use_llm_cluster_descriptions
                                    )
        
        _, text_labels = cluster_model.fit(queries)
        label_lookup = {query: label for query, label in zip(queries, text_labels)}

        def lookup_label(query):
            if query not in label_lookup:
                return "<UNK>"
            return label_lookup[query]

        df["description"] = df["query"].map(lookup_label)

        samples = list(set(text_labels))
        logger.info(f"Got samples. Number of samples: {len(samples)}")
        brands = ", ".join(brand_terms)

        prompt = PROMPT_TEMPLATE_TAXONOMY.format(samples=samples, brands=brands)

    else:

        logger.info("Using Elbow to define top ngram queries.")
        df_ngram = score_and_filter_df(df, ngram_range=ngram_range, min_df=min_df)
        logger.info(f"Got ngram frequency. Dataframe shape: {df_ngram.shape}")
        samples = list(set(df_ngram["query"].tolist()))[:settings.MAX_SAMPLES]
        logger.info(f"Got samples. Number of samples: {len(samples)}")
        brands = ", ".join(brand_terms)
        prompt = PROMPT_TEMPLATE_TAXONOMY.format(samples=samples, brands=brands)


    # Get response    
    if taxonomy_model == "palm":
        logger.info("Using Palm API.")
        response = get_palm_response(prompt)
        logger.info("Reviewing Palm's work.")
        prompt = PROMPT_TEMPLATE_TAXONOMY_REVIEW.format(taxonomy=response, brands=brands)
        reviewed_response = get_palm_response(prompt)
    elif taxonomy_model == "openai":
        logger.info("Using OpenAI API.")
        response = get_openai_response_chat(prompt)
        logger.info("Reviewing OpenAI's work.")
        prompt = PROMPT_TEMPLATE_TAXONOMY_REVIEW.format(taxonomy=response, brands=brands)
        reviewed_response = get_openai_response_chat(prompt)
    else:
        raise NotImplementedError("Platform must be 'palm' or 'openai'.")
    
    if not response or not reviewed_response:
        logger.error("No response from API.")
        return None
    
    if debug_responses:
        logger.info("Debugging responses.")
        logger.info('Initial response:')
        logger.info(response)
        logger.info('Reviewed response:')
        logger.info(reviewed_response)
    
    # Get structure
    logger.info("Getting structure.")
    structure = get_structure(reviewed_response)

    # Add categories
    logger.info("Adding categories.")

    df = df_original if len(df_original) > 0 else df

    df = add_categories_clustered(structure, df, 
                                  cluster_embeddings_model = cluster_embeddings_model,
                                  min_cluster_size = min_cluster_size,
                                  min_samples = min_samples)


    logger.info("Done.")

    return structure, df, samples




def add_categories_clustered(structure: List[str], df: pd.DataFrame, 
                             cluster_embeddings_model: Union[str, None] = None,
                             min_cluster_size: int = 5,
                             min_samples: int = 2,
                             match_col: str = "query") -> pd.DataFrame:
    
    """Add categories to dataframe."""
    texts = df[match_col].tolist()

    model = ClusterTopics(
            embedding_model = cluster_embeddings_model,
            min_cluster_size =  min_cluster_size,
            min_samples = None,
            reduction_dims = 5,
            cluster_model = "hdbscan",
            cluster_categories = structure,
            keep_outliers = True,
            n_jobs = 3,
        )


    labels, text_labels = model.fit(texts)
    label_lookup = {text: label for text, label in zip(texts, text_labels)}
    df['taxonomy'] = df[match_col].map(label_lookup)

    return df


def add_categories(taxonomy:List[str], df: pd.DataFrame) -> pd.DataFrame:
    """Add categories to dataframe."""

    logger.info("Finding Categories")

    queries = list(set(df['original_query'].tolist()))

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


