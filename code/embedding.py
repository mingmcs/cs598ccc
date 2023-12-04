import pandas as pd
from data import load_data
import toml
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis as RedisVectorStore
from langchain.document_loaders import DataFrameLoader


def create_vector_store(config=None):

    if config is None:
        with open('config.toml', 'r') as f:
            config = toml.load(f)

    log_data, log_summary = load_data(config["dataPath"])

    df = pd.DataFrame(columns=['log_data', 'log_summary', 'product'])
    for i in range(1, len(log_data) + 1):
        string_data = ""
        for j in range(0, len(log_data[str(i)])):
            string_data = string_data + log_data[str(i)][j] + "\n "
        df = pd.concat(
            [
                df,
                pd.DataFrame([{
                    'log_data': string_data,
                    'log_summary': log_summary[i - 1][0],
                    'product': config["logsSourceApplication"]
                }])
            ], ignore_index=True)

    redis_host = config["embedding"]["redis"]["hostName"]
    redis_port = config["embedding"]["redis"]["port"]

    embedding = OpenAIEmbeddings(
        deployment=config["embedding"]["deploymentUrl"],
        model=config["embedding"]["model"],
        openai_api_base=config["embedding"]["resourceEndpoint"],
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
