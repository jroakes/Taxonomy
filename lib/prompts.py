PROMPT_TEMPLATE_TAXONOMY = """As an expert in taxonomy creation, we need your assistance in developing a clear, high-level website taxonomy based on a provided list of topics. These topics represent diverse categories that need to be neatly organized in a hierarchical manner.

Here's an illustrative example to better understand the task at hand. Consider the following list of topics:

- Boys' t-shirts
- Boys' shoes, boy's sandals, Boys' sneakers
- Blue dress for teens, blue-green teenager dress
- Red girls dress
- Men's running shoes
- Boys' socks
- Tye-dye shirts for boys
- Boys' Nike court legacy shoes
- Plaid neck-tie, plaid bow-tie
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


PROMPT_TEMPLATE_TAXONOMY_REVIEW = """As a master of taxonomy creation, we need your assistance in developing a clear, high-level website taxonomy. Another member of the team has already created a taxonomy, but we need your help to review it and adjust it as needed.

Here is the taxonomy that was created:
{taxonomy}

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

Begin!
"""

PROMPT_TEMPLATE_TAXONOMY_16k = """
As an expert in taxonomy creation, we need your assistance in developing a clear, high-level website taxonomy based on a provided list of topics. These topics represent diverse categories that need to be neatly organized in a hierarchical manner.

Subject: {subject}

Topics:
{query_data}

The topics are a list of topic ngrams and their scores. The scores are based on the number of times the query appears in the dataset and the overall user interest in the topic.  Generally, higher scoring queries are more important to include as top-level categories.

Please adhere to the following format for your output:

- Category
  - Subcategory
    - Sub-subcategory
  - Subcategory
- Category
  - Subcategory
  ...

If anything can't be categorized, please add it to the Miscellaneous category. Please exclude mentioning the following brand terms in the taxonomy: {brand_terms}.  In addition, ignore any topics that are not relevant to the products and services offered by the company.

Begin!
"""



PROMPT_TEMPLATE_CLUSTER = """As an expert at understanding search intent, We need your help to provide the main subject being sought after in the following list of search queries. Please ONLY provide the subject and no other information. For example, if the search queries are 'adidas shoes, nike shoes, converse shoes', the subject is 'Name-brand shoes'.

Search Topics:
{samples}

Subject: """
