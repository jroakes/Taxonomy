"""Global settings for the project."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from loguru import logger


load_dotenv()

# Processing
MAX_WORKERS = 2
DEBUG_TEXT_FILES = False

# NLP
CROSSENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
OPEN_AI_MODEL = "gpt-4"
PALM_MODEL = "models/text-bison-001"
OPENAI_REQUEST_TIMEOUT = 500
API_RETRY_ATTEMPTS = 5
TARGET_SECTION_LEN = 1000
RANDOM_SEED = 42


# Search Console Drive
SERVICE_ACCOUNT_CREDENTIALS = "service.credentials.json"
SERVICE_ACCOUNT_SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
SERVICE_ACCOUNT_SUBJECT = "clients@locomotive.agency"


# Login
AUTHENTICATION_YAML = "auth-config.yaml"


# Environment variables. Set these at the environment level to revealing secure details.
try:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    PALM_API_KEY = os.environ["PALM_API_KEY"]
    SCRAPINGBEE_API_KEY = os.environ["SCRAPINGBEE_API_KEY"]
    ENTITIES_ENDPOINT = os.environ["ENTITIES_ENDPOINT"]
    VALUESERP_API_KEY = os.environ["VALUESERP_API_KEY"]
    SEMRUSH_API_KEY = os.environ["SEMRUSH_API_KEY"]
    DOCUMENT_CACHE_DIRECTORY = os.environ["DOCUMENT_CACHE_DIRECTORY"]
except KeyError as e:
    logger.error("Environment variable not set: {}", str(e))
