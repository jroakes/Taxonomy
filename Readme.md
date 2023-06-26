# Auto Taxonomy Creation



## Example
```
from main import create_taxonomy

taxonomy, df = create_taxonomy("https://www.domain.com/",
                    text_column = None,
                    search_volume_column = None,
                    platform = "openai", # "palm" or "openai"
                    days = 30,
                    ngram_range = (1, 6),
                    min_df = 2,
                    brand = "Brand Name",
                    limit_queries = 5)

print(taxonomy)
```

### Parameters
* `text_column`: The column with text if given a CSV or Pandas DataFrame
* `search_volume_column`: The column with search volumne if given a CSV or Pandas DataFrame
* `platform`: LLM Platform.  Uses OpenAI and Palm2
* `days`: If providing a GSC property, pulls this many days of data.  If not, does nothing.
* `ngram_range`: Breaks queries into these sets of ngrams.  Shouldn't need to change.
* `min_df`: Limits ngrams to this frequency of being found in the queries.  Can adjust larger with larger amounts of data.
* `brand`: If provided, will pull the brand terms from queries.  Helpful for sites with majority brand queries.
* `limit_queries`: For GSC only, this limits the number of queries per URL.  Good to keep indvidual posts from dominating queries.


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