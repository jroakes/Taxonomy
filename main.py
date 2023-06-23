
from typing import Union
import pandas as pd
from lib.searchconsole import load_gsc_account_data, load_available_gsc_accounts
from lib.nlp import clean_gsc_dataframe, get_ngram_frequency, merge_ngrams, filter_knee, get_structure
from lib.api import get_openai_response_chat, get_palm_response, PROMPT_TEMPLATE

from loguru import logger
import requests




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


    elif isinstance(data, pd.DataFrame):
        df = data

        if text_column is None:
            text_column = input("What is the name of the column with the queries? ")
        
        if search_volume_column is None:
            search_volume_column = input("What is the name of the column with the search volume? ")
        

        # Rename columns
        df = df.rename(columns={text_column: "query", search_volume_column: "search_volume"})

        # Remove other columns
        df = df[["query", "search_volume"]]

        # Remove non-english characters from query
        df["query"] = df["query"].str.replace(r'[^a-zA-Z0-9\s]', '')

        # Trim whitespace from query
        df["query"] = df["query"].str.strip()

        # Remove rows where query is empty
        df = df[df["query"] != '']

        # Convert search volume to int
        df["search_volume"] = df["search_volume"].astype(int)

        # Remove rows where search volume is empty or na
        df = df[df["search_volume"].notna()]

    else:
        raise ValueError("Data must be a URL string or pandas dataframe.")


    return df


def score_and_filter_df(df: pd.DataFrame,
                        ngram_range: tuple = (1, 6),
                        min_df: int = 2,) -> pd.DataFrame:
    """Score and filter dataframe."""

    df_ngram = get_ngram_frequency(df['query'].tolist(), ngram_range=ngram_range, min_df=min_df)

    df_ngram = merge_ngrams(df_ngram)

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

    return df_ngram





def create_taxonomy(data: Union[str, pd.DataFrame],
                    text_column: str = None,
                    search_volume_column: str = None,
                    platform: str = "palm", # "palm" or "openai"
                    days: int = 30,
                    ngram_range: tuple = (1, 6),
                    min_df: int = 2,
                    brand: str = None,
                    limit_queries: int = 5):
    """Kickoff function to create taxonomy from GSC data."""

    # Get data
    df = get_data(data, text_column, search_volume_column, days, brand, limit_queries)
    logger.info(f"Got Data. Dataframe shape: {df.shape}")

    # Get ngram frequency
    df_ngram = score_and_filter_df(df, ngram_range, min_df)
    logger.info(f"Got ngram frequency. Dataframe shape: {df_ngram.shape}")

    # Get samples
    samples = df_ngram["query"].tolist()[:1000]
    logger.info(f"Got samples. Number of samples: {len(samples)}")

    prompt = PROMPT_TEMPLATE.format(samples=samples, brand=brand)
    
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
    
    logger.info(response)
    
    # Get structure
    logger.info("Getting structure.")
    structure = get_structure(response)

    logger.info("Done.")

    return structure, df









