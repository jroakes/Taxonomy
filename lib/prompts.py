

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

These are brands that may be discussed in the topics: `{brands}`.
DO NOT include these brand names in your taxonomy. For example, if there is a topic about 'adidas shoes', do not add 'adidas' to your taxonomy.
DO NOT invent any sub-categories that do not naturally arise from the provided descriptions. For example, if there is no mention of or implied relationship to a 'leather' sub-category in the topics, do not add 'leather' to your taxonomy.
DO NOT create an "other" category. If there is a topic that does not fit into any of the categories, please leave it out of the taxonomy.

We expect the taxonomy to be broad rather than deeply detailed. As a rule of thumb, please keep your taxonomy no more than three levels deep.

Begin!
"""


PROMPT_TEMPLATE_TAXONOMY_LLM_DESCRIPTIONS = """As an expert at taxonomy creation, we need your help to develop a high-level taxonomy. You will be given a list of descriptions of website sections and must distill them into a clear and concise taxonomy.

As an example, here is a list of sample descriptions:
```
This section is about mens running shoes
This section is about bow ties
This section is about neck ties
Descriptions of boys shirts
various boys socks
All about girls dresses
mens shoe resources
neck ties
```

Here is how the sample topics are grouped into a taxonomy:
```
- mens
  - shoes
    - running
    - resources
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

These are brands that may be discussed in the topics: `{brands}`.
DO NOT include these brand names in your taxonomy. For example, if there is a topic about 'adidas shoes', do not add 'adidas' to your taxonomy.
DO NOT invent any sub-categories that do not naturally arise from the provided descriptions. For example, if there is no mention of or implied relationship to a 'leather' sub-category in the topics, do not add 'leather' to your taxonomy.
DO NOT create an "other" category. If there is a topic that does not fit into any of the categories, please leave it out of the taxonomy.

We expect the taxonomy to be broad rather than deeply detailed. As a rule of thumb, please keep your taxonomy no more than three levels deep.

Begin!
"""




PROMPT_TEMPLATE_CLUSTER = """As an expert at reviewing search queries, please provide a concise name for a website section of content about the queries.
Queries:
{samples}

Section Name:"""
