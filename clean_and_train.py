import datetime
import functools
import logging
import sys
from pathlib import Path

import pandas as pd

import global_options
import parse
from culture import culture_models, file_util, preprocess

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def clean_file(in_file, out_file):
    """clean the entire corpus (output from CoreNLP)
    
    Arguments:
        in_file {str or Path} -- input corpus, each line is a sentence
        out_file {str or Path} -- output corpus
    """
    a_text_clearner = preprocess.text_cleaner()
    parse.process_largefile(
        input_file=in_file,
        output_file=out_file,
        input_file_ids=[
            str(i) for i in range(file_util.line_counter(in_file))
        ],  # fake IDs (do not need IDs for this function).
        output_index_file=None,
        function_name=functools.partial(a_text_clearner.clean),
        chunk_size=200000,
    )


# clean the parsed text (remove POS tags, stopwords, etc.) ----------------
clean_file(
    in_file=Path(global_options.DATA_FOLDER, "processed", "parsed", "documents.txt"),
    out_file=Path(global_options.DATA_FOLDER, "processed", "unigram", "documents.txt"),
)


# train and apply a phrase model to detect 2-word phrases ----------------
culture_models.train_bigram_model(
    input_path=Path(
        global_options.DATA_FOLDER, "processed", "unigram", "documents.txt"
    ),
    model_path=Path(global_options.MODEL_FOLDER, "phrases", "bigram.mod"),
)
culture_models.file_bigramer(
    input_path=Path(
        global_options.DATA_FOLDER, "processed", "unigram", "documents.txt"
    ),
    output_path=Path(
        global_options.DATA_FOLDER, "processed", "bigram", "documents.txt"
    ),
    model_path=Path(global_options.MODEL_FOLDER, "phrases", "bigram.mod"),
    scoring="original_scorer",
    threshold=global_options.PHRASE_THRESHOLD,
)

# train and apply a phrase model to detect 3-word phrases ----------------
culture_models.train_bigram_model(
    input_path=Path(global_options.DATA_FOLDER, "processed", "bigram", "documents.txt"),
    model_path=Path(global_options.MODEL_FOLDER, "phrases", "trigram.mod"),
)
culture_models.file_bigramer(
    input_path=Path(global_options.DATA_FOLDER, "processed", "bigram", "documents.txt"),
    output_path=Path(
        global_options.DATA_FOLDER, "processed", "trigram", "documents.txt"
    ),
    model_path=Path(global_options.MODEL_FOLDER, "phrases", "trigram.mod"),
    scoring="original_scorer",
    threshold=global_options.PHRASE_THRESHOLD,
)

# train the word2vec model ----------------
print(datetime.datetime.now())
print("Training w2v model...")
culture_models.train_w2v_model(
    input_path=Path(
        global_options.DATA_FOLDER, "processed", "trigram", "documents.txt"
    ),
    model_path=Path(global_options.MODEL_FOLDER, "w2v", "w2v.mod"),
    size=global_options.W2V_DIM,
    window=global_options.W2V_WINDOW,
    workers=global_options.N_CORES,
    iter=global_options.W2V_ITER,
)
