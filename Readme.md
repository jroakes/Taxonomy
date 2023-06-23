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

## Cross Encoder

```
from sentence_transformers import CrossEncoder
from itertools import product
from typing import List, Tuple
import settings

def create_tuples(l1: List[str], l2: List[str]) -> List[Tuple[str, str]]:
    return [(item1, item2) for item1, item2 in product(l1, l2) if item1 != item2]

queries = list(set(df['original_query'].tolist()))
categories = taxonomy.copy()

model = CrossEncoder(settings.CROSSENCODER_MODEL_NAME, max_length=128)

sentence_pairs = create_tuples(categories, queries)

scores = model.predict(sentence_pairs, batch_size=512)

```