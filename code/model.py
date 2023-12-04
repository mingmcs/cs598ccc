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

    text = '\n'.join(input)

    prompts = [
        {
            "role": "system",
            "content": f"You are a {config['logsSourceApplication']} specialist"
        },
        {
            "role": "user",
            "content": f"Create a summary of the following {config['logsSourceApplication']} in less than 20 words."
                       "The output is should be in json format with the following keys: "
                       "summary, anomaly"
        },
        {
            "role": "assistant",
            "content": "Yes, I can do it."
        },
        {
            "role": "user",
            "content": f"create a summary for: {text}"
        }
    ]

    if vector_store is not None:
        results = vector_store.similarity_search_with_score(text, k=config["embedding"]["redis"]["kDocs"])
        for i, j in enumerate(results):
            assistant_content = j[0].metadata.get('log_summary')
            prompts.append({"role": "assistant", "content": assistant_content})

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

    return response.choices[0].message.content
