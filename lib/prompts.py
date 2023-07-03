

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

Here is how the sample topics may be grouped into a taxonomy:
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

Please provide a high-level hierarchical taxonomy that broadly represent the major themes found in the topics. 

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

To do a great job, please keep the following in mind:
* These are brands that may be discussed in the topics: `{brands}`. DO NOT include these brand names in your taxonomy. For example, if there is a topics about 'adidas shoes' and 'adidas' is one of the brands, do not add the word 'adidas' to your taxonomy.
* DO NOT invent or guess any sub-categories that do not naturally arise from the provided topics.
* DO NOT create an "other", "miscellaneous", or other catch-all category. If there is a topics that would require its own main category and it would be the only member, please ignore it.
* IGNORE login, about, contact, and other topics that are not about products or services.
* DO NOT Place items in more than one category. For example, if there is a topic about 'mens running shoes', do not add 'mens' and 'shoes' to your taxonomy at the same level. Pick one.

We expect the taxonomy to be broad rather than deeply detailed. As a rule of thumb, please keep your taxonomy no more than four levels deep, and no more than a few top-level categories.

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

Please provide a high-level hierarchical taxonomy that broadly represent the major themes found in the section desctiptions. 
 

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

To do a great job, please keep the following in mind:
* These are brands that may be discussed in the sections: `{brands}`. DO NOT include these brand names in your taxonomy. For example, if there is a section about 'adidas shoes', do not add 'adidas' to your taxonomy.
* DO NOT invent or guess any sub-categories that do not naturally arise from the provided section descriptions.
* DO NOT create an "other", "miscellaneous", or other catch-all category. If there is a section that would require its own main category and it would be the only member, please ignore it.
* IGNORE login, about, contact, and other sections that are not about products or services.
* DO NOT Place items in more than one category. For example, if there is a section description about 'mens running shoes', do not add 'mens' and 'shoes' to your taxonomy. Pick one.

We expect the taxonomy to be broad rather than deeply detailed. As a rule of thumb, please keep your taxonomy no more than four levels deep, and no more than a few top-level categories.

Begin!
"""




PROMPT_TEMPLATE_CLUSTER = """As an expert at reviewing search queries, please provide a concise name for a website section of content about the queries.
Queries:
{samples}

Section Name:"""
