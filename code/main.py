import toml
from data import load_data
from embedding import create_vector_store
import pandas as pd
from model import open_ai_summary
import time


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    with open('config.toml', 'r') as f:
        config = toml.load(f)
        print(config)

    log_file = config["dataPath"]

    log_data, log_summary = load_data(log_file)

    print(f'{config["logsSourceApplication"]} sample log:\n')
    print('\n'.join(log_data['2']))
    print(f'\n{config["logsSourceApplication"]} sample summary:\n {log_summary[1][0]}')

    engines = config["engines"]
    results = pd.DataFrame(columns=['ref'] + engines + ['log'])

    # use first 10 rows as training set
    train_logs = []
    train_summaries = []
    for i in range(5):
        train_logs.append(log_data[str(i + 1)])
        train_summaries.append(log_summary[i][0])

    vector_store = None
    if config["embedding"]["useEmbeddings"]:
        vector_store = create_vector_store(config)

    try:

        for index, ref in enumerate(log_summary):
            if index < 10 or index > 20:
                continue
            new_row = [ref]
            log_sample = log_data[str(index + 1)]
            for engine in engines:
                summary = open_ai_summary(
                    engine,
                    log_sample,
                    train_logs,
                    train_summaries,
                    vector_store=vector_store,
                    config=config
                )
                new_row.append(summary)
                time.sleep(10)

            new_row.append('\n'.join(log_sample))
            new_row = pd.DataFrame([new_row], columns=results.columns)
            results = pd.concat([results, new_row], ignore_index=True)

    except KeyboardInterrupt:
        print("Finishing early and saving results.")
    finally:
        results.to_csv(f'{config["outputDir"]}/openai_{config["logsSourceApplication"].lower()}.csv', index=False)
