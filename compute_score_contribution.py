# Compute contribution of each words (for TF, TFIDF, WFIDF)
import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import global_options
import score
from culture import culture_dictionary


def recalculate_contribution(
    documents, document_ids, all_dict_words, df_dict, N_doc, word_weights=None
):
    contribution_TF = defaultdict(int)
    contribution_WFIDF = defaultdict(int)
    contribution_TFIDF = defaultdict(int)
    contribution_TFIDF_SIMWEIGHT = defaultdict(int)
    contribution_WFIDF_SIMWEIGHT = defaultdict(int)
    for i, doc in enumerate(tqdm(documents)):
        document = doc.split()
        c = Counter(document)
        for pair in c.items():
            if pair[0] in all_dict_words:
                contribution_TF[pair[0]] += pair[1]
                w_ij = (1 + math.log(pair[1])) * math.log(N_doc / df_dict[pair[0]])
                contribution_WFIDF[pair[0]] += w_ij
                w_ij = pair[1] * math.log(N_doc / df_dict[pair[0]])
                contribution_TFIDF[pair[0]] += w_ij
                w_ij = (
                    pair[1] * word_weights[pair[0]] * math.log(N_doc / df_dict[pair[0]])
                )
                contribution_TFIDF_SIMWEIGHT[pair[0]] += w_ij
                w_ij = (
                    (1 + math.log(pair[1]))
                    * word_weights[pair[0]]
                    * math.log(N_doc / df_dict[pair[0]])
                )
                contribution_WFIDF_SIMWEIGHT[pair[0]] += w_ij
    contribution_dict = {
        "TF": contribution_TF,
        "TFIDF": contribution_TFIDF,
        "WFIDF": contribution_WFIDF,
        "TFIDF+SIMWEIGHT": contribution_TFIDF_SIMWEIGHT,
        "WFIDF+SIMWEIGHT": contribution_WFIDF_SIMWEIGHT,
    }
    return contribution_dict


def output_contribution(contribution_dict, out_file):
    """output contribution dict to Excel file
    Arguments:
        contribution_dict {word:contribution} -- a pre-calculated contribution dict for each word in expanded dictionary
        out_file {str} -- file name (Excel)
    """
    contribution_lst = []
    for dim in culture_dict:
        for w in culture_dict[dim]:
            w_dict = {}
            w_dict["dim"] = dim
            w_dict["word"] = w
            w_dict["contribution"] = contribution_dict[w]
            contribution_lst.append(w_dict)

    contribution_df = pd.DataFrame(contribution_lst)
    dim_dfs = []
    for dim in sorted(culture_dict.keys()):
        dim_df = (
            contribution_df[contribution_df.dim == dim]
            .sort_values(by="contribution", ascending=False)
            .reset_index(drop=True)
        )
        dim_df["total_contribution"] = dim_df["contribution"].sum()
        dim_df["relative_contribuion"] = dim_df["contribution"].div(
            dim_df["total_contribution"]
        )
        dim_df["cumulative_contribution"] = dim_df["relative_contribuion"].cumsum()
        dim_df = dim_df.drop(["total_contribution"], axis=1)
        dim_dfs.append(dim_df)
    pd.concat(dim_dfs, axis=1).to_csv(out_file)


if __name__ == "__main__":
    current_dict_path = str(
        str(Path(global_options.OUTPUT_FOLDER, "dict", "expanded_dict.csv"))
    )
    culture_dict, all_dict_words = culture_dictionary.read_dict_from_csv(
        current_dict_path
    )

    word_sim_weights = culture_dictionary.compute_word_sim_weights(current_dict_path)
    corpus, doc_ids, N_doc = score.load_doc_level_corpus()
    df_dict = pd.read_pickle(
        Path(global_options.OUTPUT_FOLDER, "scores", "temp", "doc_freq.pickle")
    )
    contributions_final_sample = recalculate_contribution(
        documents=corpus,
        document_ids=doc_ids,
        all_dict_words=all_dict_words,
        df_dict=df_dict,
        N_doc=N_doc,
        word_weights=word_sim_weights,
    )
    with open(
        Path(
            global_options.OUTPUT_FOLDER,
            "scores",
            "word_contributions",
            "contribution_final_sample.pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(contributions_final_sample, f)
    for weight_method in ["TF", "TFIDF", "WFIDF"]:
        output_contribution(
            contributions_final_sample[f"{weight_method}"],
            Path(
                global_options.OUTPUT_FOLDER,
                "scores",
                "word_contributions",
                f"word_contribution_{weight_method}.csv",
            ),
        )
