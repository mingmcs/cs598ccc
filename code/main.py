import toml
from data import load_data
from embedding import create_vector_store
import pandas as pd
from model import open_ai_summary
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    with open('config.toml', 'r') as f:
        config = toml.load(f)
        print(config)

    log_file = config["dataPath"]

    log_data, log_summary = load_data(log_file)
    log_summary = log_summary[:len(log_data)]
    log_data_list = list(log_data.values())
    logs_train, logs_test, log_summaries_train, log_summaries_test = \
        train_test_split(
            log_data_list,
            log_summary,
            random_state=config["data"]["splitRandomState"],
            train_size=config["data"]["trainProportion"]
        )

    print(f'{config["logsSourceApplication"]} sample log:\n')
    print('\n'.join(log_data['2']))
    print(f'\n{config["logsSourceApplication"]} sample summary:\n {log_summary[1][0]}')

    engines = config["engines"]

    vector_store = None
    if config["embedding"]["useEmbeddings"]:
        vector_store = create_vector_store(
            log_data=logs_train,
            log_summary=log_summaries_train,
            config=config
        )

    engine_cols = [[f"{engine} summary", f"{engine} anomaly"] for engine in engines]
    results = pd.DataFrame(columns=['ref'] + [item for sublist in engine_cols for item in sublist] + ['log'])

    try:
        for log_test, log_summary_test in tqdm(list(zip(logs_test, log_summaries_test))):
            new_row = [log_summary_test]
            for engine in engines:
                summary, anomaly = open_ai_summary(
                    engine,
                    log_test,
                    logs_train,
                    log_summaries_train,
                    vector_store=vector_store,
                    config=config
                )
                new_row += [summary, anomaly]
                time.sleep(10)

            new_row.append('\n'.join(log_test))
            new_row = pd.DataFrame([new_row], columns=results.columns)
            results = pd.concat([results, new_row], ignore_index=True)

    except KeyboardInterrupt:
        print("Finishing early and saving results.")
    finally:
        results.to_csv(f'{config["outputDir"]}/{config["outputFilename"]}.csv', index=False)
