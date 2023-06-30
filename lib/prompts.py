PROMPT_TEMPLATE_bak = """
As an expert at taxomomy creation, we need your help to develop a taxonomy. You will be given a list of topics and must distill them into a cogent taxonomy.

As an example, here is a list of sample topics:
```
boys t-shirts
boys shoes
blue dress for teens
red girls dress
mens running shoes
boys socks
tye-died chirts boys
boys nike court legacy shoes
plaid neck-tie
jim clark
adidas mens running shoes
bow ties
blue bow ties
adidas neo mens running shoes 8.5
```

Here is how the sample topics are grouped into a taxonomy:
```
- mens
  - shoes
    - running
  - ties
    - neck
    - bow
- boys
  - shirts
  - shoes
  - socks
- girls
  - dresses
```

Please provide a very high-level hierarchical taxonomy based on the topics. DO NOT try to create a taxonomy for every topic. Instead, create a taxonomy that represents the major themes found in the topics. 

Topics:
{samples}


Output format MUST be:
- Category
  - Subcategory
    - Sub-subcategory
  - Subcategory
- Category
  - Subcategory
  ...

The first level MUST be `{brand}`.
DO NOT attempt to make up sub-categories that are not in the topics.

Begin!
"""

PROMPT_TEMPLATE_TAXONOMY = """As an expert at taxonomy creation, we need your help to develop a high-level taxonomy. You will be given a list of topics and must distill them into a clear and concise taxonomy.

As an example, here is a list of sample topics:
```
boys t-shirts
boys shoes
blue dress for teens
red girls dress
mens running shoes
boys socks
tye-died chirts boys
boys nike court legacy shoes
plaid neck-tie
jim clark
adidas mens running shoes
bow ties
blue bow ties
adidas neo mens running shoes 8.5
```

Here is how the sample topics are grouped into a taxonomy:
```
- mens
  - shoes
    - running
  - ties
    - neck
    - bow
- boys
  - shirts
  - shoes
  - socks
- girls
  - dresses
```

Please provide a high-level hierarchical taxonomy based on the topics. This should broadly represent the major themes found in the topics, and we do not need a taxonomy for every individual topic. 

Topics:
{samples}

Your output should be structured as follows:
- Category
  - Subcategory
    - Sub-subcategory
  - Subcategory
- Category
  - Subcategory
  ...

This taxonomy is for the brand: `{brand}`. Please do not invent any sub-categories that do not naturally arise from the provided topics. For example, if there is no mention of or implied relationship to a 'leather' sub-category in the topics, do not add 'leather' to your taxonomy.

We expect the taxonomy to be broad rather than deeply detailed. As a rule of thumb, please keep your taxonomy no more than three levels deep.

Begin!
"""


PROMPT_TEMPLATE_TAXONOMY_LLM_DESCRIPTIONS = """As an expert at taxonomy creation, we need your help to develop a high-level taxonomy. You will be given a list of descriptions of user searches and must distill them into a clear and concise taxonomy.

As an example, here is a list of sample descriptions:
```
These are searches about running shoes
These are searches about bow ties
These are searches about neck ties
Searches about boys shirts
People looking for girls dresses
Searches about mens shoes
neck ties
```

Here is how the sample topics are grouped into a taxonomy:
```
- mens
  - shoes
    - running
  - ties
    - neck
    - bow
- boys
  - shirts
  - shoes
  - socks
- girls
  - dresses
```

Please provide a high-level hierarchical taxonomy based on the descriptions. This should broadly represent the major themes found in the descriptions. 

Descriptions:
{samples}

Your output should be structured as follows:
- Category
  - Subcategory
    - Sub-subcategory
  - Subcategory
- Category
  - Subcategory
  ...

This taxonomy is for the brand: `{brand}`. Please do not invent any sub-categories that do not naturally arise from the provided descriptions. For example, if there is no mention of or implied relationship to a 'leather' sub-category in the topics, do not add 'leather' to your taxonomy.

We expect the taxonomy to be broad rather than deeply detailed. As a rule of thumb, please keep your taxonomy no more than three levels deep.

Begin!
"""


PROMPT_TEMPLATE_CLUSTER_BAK = """As an expert at reviewing search queries, we need your help to understand what topics are being searched for. 
You will be given a list of search queries and must distill them into a clear and descriptive synopsis of the topics.

A good output description would be:
Users looking for mens shoes in various colors and sizes

Please provide a sentence (or two) describing what the users are looking for.

Queries:
{samples}

Begin!"""


PROMPT_TEMPLATE_CLUSTER = """As an expert at reviewing search queries, please provide a concise name for a section of content about the queries.
Queries:
{samples}

Begin!"""
