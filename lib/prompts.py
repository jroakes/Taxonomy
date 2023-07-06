

PROMPT_TEMPLATE_TAXONOMY_bak = """As an expert at taxonomy creation, we need your help to develop a high-level taxonomy. You will be given a list of topics and must distill them into a clear and concise taxonomy.

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
* These are brands that may be discussed in the topics: `{brands}`. DO NOT include these brand names in your taxonomy. For example, if there is a topics about 'adidas shoes' and 'adidas' is one of the given brands, do not add the word 'adidas' to your taxonomy name.
* DO NOT invent or guess any sub-categories that do not naturally arise from the provided topics.
* DO NOT create an catch-all or outlier category like "other" or "miscellaneous". Please ignore outlier topics.
* IGNORE login, about, contact, and other topics that are not about products or services.
* If a parent category is used for many sub-topics, consider splitting it into sub-categories or removing it.
* Make sure that category names are short but descriptive. We want to avoid lond category names, but we also want to avoid category names that are too vague.

We expect the taxonomy to be broad rather than deeply detailed. As a rule of thumb, please keep your taxonomy no more than four levels deep, and no more than a few top-level categories.

Begin!
"""

PROMPT_TEMPLATE_TAXONOMY = """As an expert in taxonomy creation, we need your assistance in developing a clear, high-level website taxonomy based on a provided list of topics. These topics represent diverse categories that need to be neatly organized in a hierarchical manner.

Here's an illustrative example to better understand the task at hand. Consider the following list of topics:

- Boys' t-shirts
- Boys' shoes
- Blue dress for teens
- Red girls dress
- Men's running shoes
- Boys' socks
- Tye-dye shirts for boys
- Boys' Nike court legacy shoes
- Plaid neck-tie
- Adidas men's running shoes
- Bow ties
- Blue bow ties
- Adidas Neo men's running shoes size 8.5

These topics could be organized into the following taxonomy:

- Men
  - Shoes
    - Running
      - Adidas
  - Ties
    - Neck
    - Bow
      - Blue
- Boys
  - Shirts
  - Shoes
    - Nike
  - Socks
- Girls
  - Dresses
    - Blue
    - Red

Now, we'd like you to apply a similar approach to organize the following list of topics:

{samples}

Please adhere to the following format for your output:

- Category
  - Subcategory
    - Sub-subcategory
  - Subcategory
- Category
  - Subcategory
  ...

In order to effectively accomplish this task, you MUST follow the following guidelines:

* Brands: The topics may mention these specific brands `{brands}`. When creating your taxonomy, please omit these brand terms. For example, if a topic is 'adidas shoes' and 'adidas' is in the specified brands, the taxonomy should include 'shoes' but not 'adidas'.
* Guessing: AVOID inventing or speculating on any subcategory subjects that are not directly reflected in the provided topics.
* Miscellaneous: Some topics are outliers, are too vague, or are not relevant to the products and services offered by the company. Assign these topics to a top-level category called 'Miscellaneous'. Here are some examples of topics that should be labeled as 'Miscellaneous':
  * 'Jeans' is only mentioned in a few random topics.
  * 'llc' is a topic and no category about business incorporation exists.
  * 'about'-type topics.
  * 'contact'-type topics.
  * 'jim h' is a topic and is not a notable person.
  * 'login'-type topics.
  * 'sign up'-type topics.
* Depth of Taxonomy: The taxonomy should be no more than four levels deep (i.e., Category > Subcategory > Sub-subcategory > Sub-sub-subcategory) and should have only a few top-level categories. The taxonomy should be broad rather than deeply detailed. Topics should be categorized into subjects and subjects into cogent categories.
* Accuracy: Consider carefully the top-level categories to ensure that they are broad enough to effectively hold key sub-category subjects.
* Readability: Ensure that category names are concise yet descriptive.
* Duplication: DO NOT assign a topic to 'Miscellaneous' if a similar subject has already been assigned as a child of another category.

Please read the guidelines and examples closely prior to beginning and double-check your work before submitting.

Begin!
"""


PROMPT_TEMPLATE_TAXONOMY_LLM_DESCRIPTIONS = """As an expert in taxonomy creation, we need your assistance in developing a clear, high-level website taxonomy based on a provided list of section description topics. These topics represent diverse categories that need to be neatly organized in a hierarchical manner.

Here's an illustrative example to better understand the task at hand. Consider the following list of topics:

- Section about Boys' and Girls t-shirts
- Boys' shoes content
- Section on Blue dress for teens
- Content on Red girls dress
- Men's running shoes category
- Boys' socks searches
- Tye-dye shirts for boys
- Boys' Nike court legacy shoes
- Plaid neck-tie resources
- Adidas men's running shoes products
- Bow ties examples
- Blue bow ties content
- Adidas Neo men's running shoes category

After careful analysis and categorization, these topics could be organized into the following taxonomy:

- Men
  - Shoes
    - Running
      - Adidas
  - Ties
    - Neck
    - Bow
      - Blue
- Boys
  - Shirts
    - t-shirts
  - Shoes
  - Socks
- Girls
  - Shirts
    - t-shirts
  - Dresses

Now, we'd like you to apply a similar approach to organize the following list of topics:

{samples}

Please adhere to the following format for your output:

- Category
  - Subcategory
    - Sub-subcategory
  - Subcategory
- Category
  - Subcategory
  ...

In order to effectively accomplish this task, please follow the following guidelines:

* Brands: The topics may mention these specific brands `{brands}`. When creating your taxonomy, please omit these brand terms. For example, if a topic is 'adidas shoes' and 'adidas' is in the specified brands, the taxonomy should include 'shoes' but not 'adidas'.
* Guessing: AVOID inventing or speculating on any subcategory subjects that are not directly reflected in the provided topics.
* Miscellaneous: Some topics are outliers, are too vague, or are not relevant to the products and services offered by the company. Assign these topics to a top-level category called 'Miscellaneous'. Here are some examples of topics that should be labeled as 'Miscellaneous':
  * 'Jeans' is only mentioned in a few random topics.
  * 'llc' is a topic and no category about business incorporation exists.
  * 'about'-type topics.
  * 'contact'-type topics.
  * 'jim h' is a topic and is not a notable person.
  * 'login'-type topics.
  * 'sign up'-type topics.
* Depth of Taxonomy: The taxonomy should be no more than four levels deep (i.e., Category > Subcategory > Sub-subcategory > Sub-sub-subcategory) and should have only a few top-level categories. The taxonomy should be broad rather than deeply detailed. Topics should be categorized into subjects and subjects into cogent categories.
* Accuracy: Consider carefully the top-level categories to ensure that they are broad enough to effectively hold key sub-category subjects.
* Readability: Ensure that category names are concise yet descriptive.
* Duplication: DO NOT assign a topic to 'Miscellaneous' if a similar subject has already been assigned as a child of another category.

Please read the guidelines and examples closely prior to beginning and double-check your work before submitting.
DO NOT assign a topic to 'Miscellaneous' if it has already been assigned as a child of another category.

Begin!
"""

PROMPT_TEMPLATE_TAXONOMY_REVIEW = """As a master of taxonomy creation, we need your assistance in developing a clear, high-level website taxonomy. Another member of the team has already created a taxonomy, but we need your help to review it and adjust it as needed.

Here is the taxonomy that was created:
```{taxonomy}```

Please review the taxonomy and make any necessary changes. If you believe that the taxonomy is correct, please submit it as is.

Here are some guidelines for reviewing the taxonomy:
* Remove any Miscellaneous sub-categories that are already assigned to other categories. For example, if there is a category called 'Nike > Shoes > NEO Vulc' and another category called 'Miscellaneous > Neo Vulc', please remove the 'Miscellaneous > Neo Vulc' category.
* Make sure all category designations are accurate and appropriate.
* Ensure that categories are not duplicated. For example, if there is a category called 'Nike > Shoes > NEO Vulc' and another category called 'Nike > Shoes > NEO Vulc > NEO Vulc', please remove the 'Nike > Shoes > NEO Vulc > NEO Vulc' category.
* Review the category names for readability. Ensure that category names are concise yet descriptive.


Please keep the formatting of the original taxonomy. The taxonomy should be structured as follows:
- Category
  - Subcategory
    - Sub-subcategory
  - Subcategory
  . . .
- Category

Please read the guidelines closely prior to beginning and double-check your work before submitting.

Begin!"""



PROMPT_TEMPLATE_CLUSTER = """As an expert at understanding search intent, We need your help to provide the main subject being sought after in the following list of topics. You will be given a list of topics and must distill them into a clear and concise subject.

Topics:
{samples}


Subject:"""
