"""API functions for the app."""
from __future__ import annotations

from typing import List, Tuple, Union
import json

import openai
from google.generativeai.types import safety_types
import google.generativeai as palm

import settings
from scrapingbee import ScrapingBeeClient
from loguru import logger


from tenacity import (
    retry,
    RetryError,
    stop_after_attempt,
    wait_random_exponential,
)

import settings

# Set OpenAI API key
openai.api_key = settings.OPENAI_API_KEY

# Set Palm API key
palm.configure(api_key='AIzaSyBQIlQ1KBpbvabUepv51I15YJLnnUe8VJM')




READABILITY_JS = """
(async function main () {

  let page_content = async()=>{

    const readability = await import('https://cdn.skypack.dev/@mozilla/readability');

    let documentClone = document.cloneNode(true);
    let rd = new readability.Readability(documentClone)
    rd._clean_b = rd._clean
    rd._clean = function(e, tag) {
        let allowed = ['button', 'a']
        return allowed.indexOf(tag) ? this._clean_b(e, tag) > -1 : true;
    }
    let rd_result = rd.parse();
    return rd_result.textContent;
  }

  let content = await page_content();
  return content;
    
})();
"""

HEADERS = {
    "DNT": "1",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.50",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}


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

PROMPT_TEMPLATE = """As an expert at taxonomy creation, we need your help to develop a high-level taxonomy. You will be given a list of topics and must distill them into a clear and concise taxonomy.

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





def get_url_content_scrapingbee(url: str) -> Union[str, None]:
    """Get the content of a URL using ScrapingBee."""

    client = ScrapingBeeClient(api_key=settings.SCRAPINGBEE_API_KEY)
    response = client.get(
        url,
        headers=HEADERS,
        params={
            "block_ads": "True",
            "json_response": "True",
            "wait_browser": "load",
            "js_scenario": {
                "strict": False,
                "instructions": [{"evaluate": READABILITY_JS}],
            },
        },
    )

    if response.status_code == 200:
        url_data = response.json()

        if isinstance(url_data, dict) and "evaluate_results" in url_data:
            content = url_data["evaluate_results"]
            if isinstance(content, list) and len(content) > 0:
                return content[0]
            elif isinstance(content, str) and len(content) > 0:
                return content

        if isinstance(url_data, dict) and "js_scenario_report" in url_data:
            report = url_data["js_scenario_report"]
            logger.error(f"ScrapingBee error: {json.dumps(report, indent=2)}")

    logger.error(f"ScrapingBee error: {response.status_code}")

    return None



class APIError(Exception):
    """Base class for API errors."""

    pass


class OpenAIError(APIError):
    """Error for OpenAI API."""

    pass


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(settings.API_RETRY_ATTEMPTS),
)
def get_openai_response(
    messages: List[dict], model: str = "gpt-3.5-turbo"
) -> Union[str, None]:
    """Get a response from OpenAI's API."""

    logger.info("Getting OpenAI response...")
    try:
        chat_completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            request_timeout=settings.OPENAI_REQUEST_TIMEOUT,
            max_tokens=2000,
            temperature=0.0,
            n=1,
        )

    except Exception as e:
        logger.error("OpenAI API Error: " + str(e))
        logger.info("Messages: " + str(messages))
        raise OpenAIError(str(e))

    return chat_completion.choices[0].message.content


def get_openai_response_chat(
    messages: List[dict] | str,
    model: str = settings.OPEN_AI_MODEL,
    system_message: dict | str = "You are an expert taxonomy creator.",
) -> Union[str, None]:
    """Get a response from OpenAI's chat API."""

    system_message = {"role": "system", "content": system_message}

    if isinstance(messages, str):
        messages = [system_message, {"role": "user", "content": messages}]
    else:
        messages = [system_message] + messages

    try:
        return get_openai_response(messages, model=model)

    except RetryError as e:
        logger.error("API Retry Error: " + str(e))
        raise APIError(str(e))



def get_palm_response(prompt: str, model: str = settings.PALM_MODEL) -> str:

    safety_settings = [
        {
            "category": getattr(
                safety_types.HarmCategory, f"HARM_CATEGORY_{category}"
            ),
            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        }
        for category in [
                "DEROGATORY",
                "TOXICITY",
                "SEXUAL",
                "VIOLENCE",
                "DANGEROUS",
                "MEDICAL",
            ]
    ]

    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0,
        max_output_tokens=1024,
        safety_settings=safety_settings
    )

    return completion.result



