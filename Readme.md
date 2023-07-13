# Auto Taxonomy Creation



## Example with CSV
```
from main import create_taxonomy

filename = "domain_data.csv"
brand_terms = ['brand', 'brand', 'brand']

taxonomy, df, samples = create_taxonomy(filename,
                                        text_column = "keyword",
                                        search_volume_column = "search_volume",
                                        ngram_range = (1, 5),
                                        min_df = 5,
                                        brand_terms = brand_terms)

df.to_csv("taxonomy.csv", index=False)

print("\n".join(taxonomy))
```

## Example with GSC
```
from main import create_taxonomy

brand_terms = ["brand"]
property = "sc-domain:domain.com"

taxonomy, df, samples = create_taxonomy(property,
                                        days = 30,
                                        ngram_range = (1, 6),
                                        min_df = 2,
                                        brand_terms = brand_terms,
                                        limit_queries = 5)


df.to_csv("domain_taxonomy.csv", index=False)

df.head()
```



### Parameters
* `data` (Union[str, pd.DataFrame]): GSC Property, CSV Filename, or pandas dataframe.
* `text_column` (str, optional): Name of the column with the queries. Defaults to None.
* `search_volume_column` (str, optional): Name of the column with the search volume. Defaults to None.
* `cluster_embeddings_model` (Union[str, None], optional): Name of the cluster embeddings model. Defaults to None.
* `min_cluster_size` (int, optional): Minimum cluster size to use for clustering. Defaults to 5.
* `min_samples` (int, optional): Minimum samples to use for clustering. Defaults to 2.
* `days` (int, optional): Number of days to get data from. Defaults to 30.
* `ngram_range` (tuple, optional): Ngram range to use for scoring. Defaults to (1, 6).
* `min_df` (int, optional): Minimum document frequency to use for scoring. Defaults to 2.
* `brand_terms` (List[str], optional): List of brand terms to remove from queries. Defaults to None.
* `limit_queries_per_page` (int, optional): Number of queries per page to use for clustering. Defaults to 5.
* `debug_responses` (bool, optional): Whether to print debug responses. Defaults to False.




## Cross Encoder

```
from main import create_taxonomy, add_categories

filename = "data.csv"
brand = "Brand"

taxonomy, df = create_taxonomy(filename,
                    text_column = "keyword",
                    search_volume_column = "search_volume",
                    platform = "openai", # "palm" or "openai"
                    days = 30,
                    S=200,
                    ngram_range = (1, 5),
                    min_df = 1,
                    brand = None,
                    limit_queries = 1)


# Cross Encodes queries back to categories
df = add_categories(taxonomy, df, brand) 

df.to_csv("cais_taxonomy.csv", index=False)

df.head()

```