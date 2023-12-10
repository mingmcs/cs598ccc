import pandas as pd
import toml
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores.redis import Redis as RedisVectorStore
from langchain.document_loaders import DataFrameLoader


def create_vector_store(log_data, log_summary, config=None):

    if config is None:
        with open('config.toml', 'r') as f:
            config = toml.load(f)

    df = pd.DataFrame(columns=['log_data', 'log_summary', 'product'])

    for log, summaries in zip(log_data, log_summary):
        first_summary = summaries[0]
        string_data = "\n".join(log) + "\n"
        df = pd.concat(
            [
                df,
                pd.DataFrame([{
                    'log_data': string_data,
                    'log_summary': first_summary,
                    'product': config["logsSourceApplication"]
                }])
            ], ignore_index=True)

    redis_host = config["embedding"]["redis"]["hostName"]
    redis_port = config["embedding"]["redis"]["port"]

    embedding = AzureOpenAIEmbeddings(
        azure_deployment=config["embedding"]["deployment"],
        model=config["embedding"]["model"],
        azure_endpoint=config["embedding"]["resourceEndpoint"],
        openai_api_type=config["embedding"]["openAiApiType"],
        openai_api_key=config["embedding"]["apiKey"],
        openai_api_version=config["embedding"]["openAiApiVersion"],
        chunk_size=config["embedding"]["chunkSize"]
    )

    loader = DataFrameLoader(df, page_content_column="log_data")
    logs_list = loader.load()
    redis_url = f"redis://{redis_host}:{redis_port}"

    vectorstore = RedisVectorStore.from_documents(
        documents=logs_list,
        embedding=embedding,
        index_name=config["embedding"]["redis"]["indexName"],
        redis_url=redis_url
    )

    return vectorstore
