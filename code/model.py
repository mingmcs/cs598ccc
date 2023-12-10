import toml
from openai import AzureOpenAI


def open_ai_summary(model, input, logs=None, log_summaries=None, vector_store=None, config=None):
    if config is None:
        with open('config.toml', 'r') as f:
            config = toml.load(f)

    client = AzureOpenAI(
        azure_endpoint="https://aiops23.openai.azure.com/",
        api_key=config["openAiApiKey"],
        api_version="2023-09-15-preview"
    )

    user_input_log = '\n'.join(input)

    prompts = [
        {
            "role": "system",
            "content": f"You are a {config['logsSourceApplication']} specialist"
        }
    ]

    if vector_store is not None:
        results = vector_store.similarity_search_with_score(user_input_log, k=config["embedding"]["redis"]["kDocs"])
        for i, j in enumerate(results):
            vector_store_log = j[0].page_content
            vector_store_log_summary = j[0].metadata.get('log_summary')
            prompts.append(
                {
                    "role": "user",
                    "content": f"Create a summary for the following {config['logsSourceApplication']} log in less " +
                               f"than 20 words {vector_store_log}"
                }
            )
            prompts.append(
                {
                    "role": "assistant",
                    "content": vector_store_log_summary
                }
            )

    prompts.append(
        {
            "role": "user",
            "content": f"Create a summary for the following {config['logsSourceApplication']} log in less " +
                       f"than 20 words {user_input_log}"
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=prompts,
        temperature=config["model"]["temperature"],
        max_tokens=config["model"]["maxTokens"],
        top_p=config["model"]["topP"],
        frequency_penalty=config["model"]["frequencyPenalty"],
        presence_penalty=config["model"]["presencePenalty"],
        stop=None
    )

    summary = response.choices[0].message.content.replace('\n', '')

    prompts.append(
        {
            "role": "assistant",
            "content": summary
        }
    )

    prompts.append(
        {
            "role": "user",
            "content": "Were there any anomalies in the most recent log?"
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=prompts,
        temperature=config["model"]["temperature"],
        max_tokens=config["model"]["maxTokens"],
        top_p=config["model"]["topP"],
        frequency_penalty=config["model"]["frequencyPenalty"],
        presence_penalty=config["model"]["presencePenalty"],
        stop=None
    )

    anomaly = response.choices[0].message.content.replace('\n', '')

    return summary, anomaly
