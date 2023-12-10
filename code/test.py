import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize

from bert_score import BERTScorer, score
from nubia_score import Nubia

import toml
from tqdm import tqdm

from IPython.display import display

with open('config.toml', 'r') as f:
    config = toml.load(f)
    print(config)


def bleu_score(ref, gen):
    smoother = SmoothingFunction()
    return sentence_bleu([ref.split()], gen.split(), smoothing_function=smoother.method4)


def rouge_score(ref, gen):
    return rouge_scorer \
        .RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) \
        .score(ref, gen)["rougeL"] \
        .fmeasure


def meteor_score(ref, gen):
    return single_meteor_score(word_tokenize(ref), word_tokenize(gen))


def bert_score(ref, gen):
    p, r, f1 = BERTScorer(model_type='bert-base-uncased').score([gen], [ref])
    return f1.item()


def bleurt_score(ref, gen):
    p, r, f1 = score([gen], [ref], lang="en", verbose=False)
    return f1.item()


def nubia_score(ref, gen):
    return Nubia().score(ref, gen, verbose=False, get_features=True)['nubia_score']


def process_dataset(file):
    df = pd.read_csv(file)

    refs = [ref.replace("['", "").replace("']", "") for ref in df["ref"]]
    df["ref"] = refs

    bleu_scores = []
    rouge_scores = []
    meteor_scores = []
    bert_scores = []
    bleurt_scores = []
    nubia_scores = []
    for summary, ref_log in tqdm(list(zip(refs, df["log"]))):
        bleu_scores.append(bleu_score(ref_log, summary))
        rouge_scores.append(rouge_score(ref_log, summary))
        meteor_scores.append(meteor_score(ref_log, summary))
        bert_scores.append(bert_score(ref_log, summary))
        bleurt_scores.append(bleurt_score(ref_log, summary))
        # nubia_scores.append(nubia_score(ref_log, summary))

    df[f"ref BLEU"] = bleu_scores
    df[f"ref ROUGE"] = rouge_scores
    df[f"ref METEOR"] = meteor_scores
    df[f"ref BERT"] = bert_scores
    df[f"ref BLEURT"] = bleurt_scores
    df[f"ref NUBIA"] = nubia_scores

    for engine in config["engines"]:
        summaries = df[f"{engine} summary"]
        ref_summaries = df["ref"]
        ref_logs = df["log"]

        bleu_scores = []
        rouge_scores = []
        meteor_scores = []
        bert_scores = []
        bleurt_scores = []
        nubia_scores = []
        for summary, ref_summary, ref_log in tqdm(list(zip(summaries, ref_summaries, ref_logs))):
            bleu_scores.append(bleu_score(ref_log, summary))
            rouge_scores.append(rouge_score(ref_log, summary))
            meteor_scores.append(meteor_score(ref_log, summary))
            bert_scores.append(bert_score(ref_log, summary))
            bleurt_scores.append(bleurt_score(ref_log, summary))
            nubia_scores.append(nubia_score(ref_log, summary))

        df[f"{engine} BLEU"] = bleu_scores
        df[f"{engine} ROUGE"] = rouge_scores
        df[f"{engine} METEOR"] = meteor_scores
        df[f"{engine} BERT"] = bert_scores
        df[f"{engine} BLEURT"] = bleurt_scores
        df[f"{engine} NUBIA"] = nubia_scores
    return df


if __name__ == '__main__':
    data_file = f"{config['outputDir']}/zookeeper_with_embeddings.csv"
    df_with_embeddings = process_dataset(data_file)
