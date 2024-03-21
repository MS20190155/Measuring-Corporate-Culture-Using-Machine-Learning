import datetime
import itertools
import os
from pathlib import Path
from functools import partial
from stanfordnlp.server import CoreNLPClient

import global_options as global_options
from culture import file_util, preprocess


def process_line(corpus_preprocessor, line, lineID):
    """Process each line and return a tuple of sentences, sentence_IDs,

    Arguments:
        line {str} -- a document
        lineID {str} -- the document ID

    Returns:
        str, str -- processed document with each sentence in a line,
                    sentence IDs with each in its own line: lineID_0 lineID_1 ...
    """
    sentences_processed = []
    doc_sent_ids = []
    try:
        sentences_processed, doc_sent_ids = corpus_preprocessor.process_document(
            line, lineID
        )
    except Exception as e:
        print(e)
        print("Exception in line: {}".format(lineID))
    return "\n".join(sentences_processed), "\n".join(doc_sent_ids)


def process_largefile(
    input_file,
    output_file,
    input_file_ids,
    output_index_file,
    function_name,
    chunk_size=100,
    start_index=None,
):
    """A helper function that transforms an input file + a list of IDs of each line (documents + document_IDs) to two output files (processed documents + processed document IDs) by calling function_name on chunks of the input files. Each document can be decomposed into multiple processed documents (e.g. sentences).
    Supports parallel with Pool.

    Arguments:
        input_file {str or Path} -- path to a text file, each line is a document
        ouput_file {str or Path} -- processed linesentence file (remove if exists)
        input_file_ids {str]} -- a list of input line ids
        output_index_file {str or Path} -- path to the index file of the output
        function_name {callable} -- A function that processes a list of strings, list of ids and return a list of processed strings and ids.
        chunk_size {int} -- number of lines to process each time, increasing the default may increase performance
        start_index {int} -- line number to start from (index starts with 0)

    Writes:
        Write the ouput_file and output_index_file
    """
    try:
        if start_index is None:
            # if start from the first line, remove existing output file
            # else append to existing output file
            os.remove(str(output_file))
            os.remove(str(output_index_file))
    except OSError:
        pass
    assert file_util.line_counter(input_file) == len(
        input_file_ids
    ), "Make sure the input file has the same number of rows as the input ID file. "

    with open(input_file, newline="\n", encoding="utf-8", errors="ignore") as f_in:
        line_i = 0
        # jump to index
        if start_index is not None:
            # start at start_index line
            for _ in range(start_index):
                next(f_in)
            input_file_ids = input_file_ids[start_index:]
            line_i = start_index
        for next_n_lines, next_n_line_ids in zip(
            itertools.zip_longest(*[f_in] * chunk_size),
            itertools.zip_longest(*[iter(input_file_ids)] * chunk_size),
        ):
            line_i += chunk_size
            print(datetime.datetime.now())
            print(f"Processing line: {line_i}.")
            next_n_lines = list(filter(None.__ne__, next_n_lines))
            next_n_line_ids = list(filter(None.__ne__, next_n_line_ids))
            output_lines = []
            output_line_ids = []
            # Use parse_parallel.py to speed things up
            for output_line, output_line_id in map(
                function_name, next_n_lines, next_n_line_ids
            ):
                output_lines.append(output_line)
                output_line_ids.append(output_line_id)

            output_lines = "\n".join(output_lines) + "\n"
            output_line_ids = "\n".join(output_line_ids) + "\n"
            with open(output_file, "a", newline="\n") as f_out:
                f_out.write(output_lines)
            if output_index_file is not None:
                with open(output_index_file, "a", newline="\n") as f_out:
                    f_out.write(output_line_ids)


def sequential_parse():
    with CoreNLPClient(
        properties={
            "ner.applyFineGrained": "false",
            "annotators": "tokenize, ssplit, pos, lemma, ner, depparse",
        },
        memory=global_options.RAM_CORENLP,
        threads=global_options.N_CORES,
        timeout=12000000,
        max_char_length=1000000,
    ) as client:
        corpus_preprocessor = preprocess.preprocessor(client)
        in_file = Path(global_options.DATA_FOLDER, "input", "documents.txt")
        in_file_index = file_util.file_to_list(
            Path(global_options.DATA_FOLDER, "input", "document_ids.txt")
        )
        out_file = Path(
            global_options.DATA_FOLDER, "processed", "parsed", "documents.txt"
        )
        output_index_file = Path(
            global_options.DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt"
        )
        process_largefile(
            input_file=in_file,
            output_file=out_file,
            input_file_ids=in_file_index,
            output_index_file=output_index_file,
            function_name=partial(
                process_line, corpus_preprocessor
            ),  # NB: I am using a Partial function for now! I will remove later with an OO design
            chunk_size=global_options.PARSE_CHUNK_SIZE,
        )
