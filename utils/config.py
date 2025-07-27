from dotenv import load_dotenv

import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, AzureOpenAI

load_dotenv()


def get_llm():
    """
    Returns an instance of the ChatOpenAI LLM with the model specified in the environment variable.
    Defaults to 'gpt-3.5-turbo' if no model is specified.
    """
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    azure_embedding_deployment_name = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

    if not azure_openai_api_key:
        raise ValueError("AZURE_OPENAI_API_KEY is not set in the environment variables for LLM.")

    llm = AzureChatOpenAI(
        api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        azure_deployment=azure_deployment_name,
        temperature=0.2,
        max_tokens=150
    )

    return llm


def get_embedding():
    """
    Returns an instance of the embedding with the model specified in the environment variable.
    Defaults to 'gpt-3.5-turbo' if no model is specified.
    """
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_embedding_api_version = os.getenv("AZURE_EMBEDDING_API_VERSION")
    azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    azure_embedding_deployment_name = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

    if not azure_openai_api_key:
        raise ValueError("AZURE_OPENAI_API_KEY is not set in the environment variables for embedding model.")

    client = AzureOpenAIEmbeddings(
        api_key=azure_openai_api_key,
        azure_endpoint=azure_openai_endpoint,
        api_version=azure_embedding_api_version,
        # dimensions=embedding_vector_dimensions,
        model=azure_embedding_deployment_name,
        timeout=None,
        max_retries=2,
    )

    return client


def get_embedding2():
    """
    Returns an instance of the embedding with the model specified in the environment variable.
    Defaults to 'gpt-3.5-turbo' if no model is specified.
    """
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_embedding_api_version = os.getenv("AZURE_EMBEDDING_API_VERSION")
    azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    azure_embedding_deployment_name = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

    if not azure_openai_api_key:
        raise ValueError("AZURE_OPENAI_API_KEY is not set in the environment variables for embedding model.")

    client = AzureOpenAI(
        api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        azure_deployment=azure_embedding_deployment_name,
    )

    return client
