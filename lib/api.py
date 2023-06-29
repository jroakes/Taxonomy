"""API functions for the app."""
from __future__ import annotations

from typing import List, Tuple, Union
import json
import numpy as np

import openai
from google.generativeai.types import safety_types
import google.generativeai as palm

from tqdm.auto import tqdm
import settings
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



def get_openai_embeddings(texts: List[str], 
                        model: str = settings.OPENAI_EMBEDDING_MODEL) -> np.ndarray:
  """Get embeddings from OpenAI's API."""

  @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
  def get_single_embedding(text: str) -> np.ndarray:
    return np.asarray(openai.Embedding.create(input = [text], model=model)['data'][0]['embedding'])

  return np.asarray([get_single_embedding(text) for text in tqdm(texts, desc="Getting OpenAI embeddings")])



def get_palm_embeddings(texts: List[str], 
                        model: str = settings.PALM_EMBEDDING_MODEL) -> np.ndarray:
  """Get embeddings from PALM's API."""

  @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
  def get_single_embedding(text: str) -> np.ndarray:
    return np.asarray(palm.generate_embeddings(model, text))

  return np.asarray([get_single_embedding(text) for text in tqdm(texts, desc="Getting PALM embeddings")])



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



