import itertools
import os
import pickle
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import pandas as pd
from tqdm import tqdm as tqdm

import global_options
from culture import culture_dictionary, file_util

# @TODO: The scoring functions are not memory friendly. The entire pocessed corpus needs to fit in the RAM. Rewrite a memory friendly version.


def construct_doc_level_corpus(sent_corpus_file, sent_id_file):
    """Construct document level corpus from sentence level corpus and write to disk.
    Dump "corpus_doc_level.pickle" and "doc_ids.pickle" to Path(global_options.OUTPUT_FOLDER, "scores", "temp"). 
    
    Arguments:
        sent_corpus_file {str or Path} -- The sentence corpus after parsing and cleaning, each line is a sentence
        sent_id_file {str or Path} -- The sentence ID file, each line correspond to a line in the sent_co(docID_sentenceID)
    
    Returns:
        [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
    """
    print("Constructing doc level corpus")
    # sentence level corpus
    sent_corpus = file_util.file_to_list(sent_corpus_file)
    sent_IDs = file_util.file_to_list(sent_id_file)
    assert len(sent_IDs) == len(sent_corpus)
    # doc id for each sentence
    doc_ids = [x.split("_")[0] for x in sent_IDs]
    # concat all text from the same doc
    id_doc_dict = defaultdict(lambda: "")
    for i, id in enumerate(doc_ids):
        id_doc_dict[id] += " " + sent_corpus[i]
    # create doc level corpus
    corpus = list(id_doc_dict.values())
    doc_ids = list(id_doc_dict.keys())
    assert len(corpus) == len(doc_ids)
    with open(
        Path(global_options.OUTPUT_FOLDER, "scores", "temp", "corpus_doc_level.pickle"),
        "wb",
    ) as out_f:
        pickle.dump(corpus, out_f)
    with open(
        Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_ids.pickle"), "wb"
    ) as out_f:
        pickle.dump(doc_ids, out_f)
    N_doc = len(corpus)
    return corpus, doc_ids, N_doc


def calculate_df(corpus):
    """Calcualte and dump a document-freq dict for all the words.
    
    Arguments:
        corpus {[str]} -- a list of documents
    
    Returns:
        {dict[str: int]} -- document freq for each word
    """
    print("Calculating document frequencies.")
    # document frequency
    df_dict = defaultdict(int)
    for doc in tqdm(corpus):
        doc_splited = doc.split()
        words_in_doc = set(doc_splited)
        for word in words_in_doc:
            df_dict[word] += 1
    # save df dict
    with open(
        Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle"), "wb"
    ) as f:
        pickle.dump(df_dict, f)
    return df_dict


def load_doc_level_corpus():
    """load the corpus constructed by construct_doc_level_corpus()
    
    Returns:
        [str], [str], int -- a tuple of a list of documents, a list of document IDs, and the number of documents
    """
    print("Loading document level corpus.")
    with open(
        Path(global_options.OUTPUT_FOLDER, "scores", "temp", "corpus_doc_level.pickle"),
        "rb",
    ) as in_f:
        corpus = pickle.load(in_f)
    with open(
        Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_ids.pickle"), "rb"
    ) as in_f:
        doc_ids = pickle.load(in_f)
    assert len(corpus) == len(doc_ids)
    N_doc = len(corpus)
    return corpus, doc_ids, N_doc


def score_tf(documents, doc_ids, expanded_dict):
    """
    Score documents using term freq. 
    """
    print("Scoring using Term-freq (tf).")
    score = culture_dictionary.score_tf(
        documents=documents,
        document_ids=doc_ids,
        expanded_words=expanded_dict,
        n_core=global_options.N_CORES,
    )
    score.to_csv(
        Path(global_options.OUTPUT_FOLDER, "scores", "scores_TF.csv"), index=False
    )


def score_tf_idf(documents, doc_ids, N_doc, method, expanded_dict, **kwargs):
    """Score documents using tf-idf and its variations
    
    Arguments:
        documents {[str]} -- list of documents
        doc_ids {[str]} -- list of document IDs
        N_doc {int} -- number of documents
        method {str} -- 
            TFIDF: conventional tf-idf 
            WFIDF: use wf-idf log(1+count) instead of tf in the numerator
            TFIDF/WFIDF+SIMWEIGHT: using additional word weights given by the word_weights dict
        expanded_dict {dict[str, set(str)]} -- expanded dictionary
    """
    if method == "TF":
        print("Scoring TF.")
        score_tf(documents, doc_ids, expanded_dict)
    else:
        print("Scoring TF-IDF.")
        # load document freq
        df_dict = pd.read_pickle(
            Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle")
        )
        # score tf-idf
        score, contribution = culture_dictionary.score_tf_idf(
            documents=documents,
            document_ids=doc_ids,
            expanded_words=expanded_dict,
            df_dict=df_dict,
            N_doc=N_doc,
            method=method,
            **kwargs
        )
        # save the document level scores (without dividing by doc length)
        score.to_csv(
            str(
                Path(
                    global_options.OUTPUT_FOLDER,
                    "scores",
                    "scores_{}.csv".format(method),
                )
            ),
            index=False,
        )
        # save word contributions
        pd.DataFrame.from_dict(contribution, orient="index").to_csv(
            Path(
                global_options.OUTPUT_FOLDER,
                "scores",
                "word_contributions",
                "word_contribution_{}.csv".format(method),
            )
        )


if __name__ == "__main__":
    current_dict_path = str(
        str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv"))
    )
    culture_dict, all_dict_words = culture_dictionary.read_dict_from_csv(
        current_dict_path
    )
    # words weighted by similarity rank (optional)
    word_sim_weights = culture_dictionary.compute_word_sim_weights(current_dict_path)

    ## Pre-score ===========================
    # aggregate processed sentences to documents
    corpus, doc_ids, N_doc = construct_doc_level_corpus(
        sent_corpus_file=Path(
            global_options.DATA_FOLDER, "processed", "trigram", "documents.txt"
        ),
        sent_id_file=Path(
            global_options.DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt"
        ),
    )
    word_doc_freq = calculate_df(corpus)

    ## Score ========================
    # create document scores
    methods = ["TF", "TFIDF", "WFIDF"]
    for method in methods:
        score_tf_idf(
            corpus,
            doc_ids,
            N_doc,
            method=method,
            expanded_dict=culture_dict,
            normalize=False,
            word_weights=word_sim_weights,
        )
