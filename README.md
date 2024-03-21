# app_corporate_culture_measurer
![build status](https://github.com/praisetompane/app_corporate_culture_measurer/actions/workflows/app_corporate_culture_measurer.yaml/badge.svg)

## Introduction
restructure [Measuring Corporate Culture Using Machine Learning](https://github.com/MS20190155/Measuring-Corporate-Culture-Using-Machine-Learning).
Kai Li, Feng Mai, Rui Shen, Xinyan Yan, [__Measuring Corporate Culture Using Machine Learning__](https://academic.oup.com/rfs/advance-article-abstract/doi/10.1093/rfs/hhaa079/5869446?redirectedFrom=fulltext), _The Review of Financial Studies_, 2020; DOI:[10.1093/rfs/hhaa079](http://dx.doi.org/10.1093/rfs/hhaa079) 
[[Available at SSRN]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3256608)

objectives:
- restructure into REST API driven application. 
- accept aribitary corpus. 
- generate an aribitary number of culture frameworks.
- explore Word2Vec:
    - in the context of an application.
    - how word vector representations are learned.
    - references:
        - https://code.google.com/archive/p/word2vec/
        - https://www.askpython.com/python-modules/gensim-word2vec
- generate culture frameworks based on various corpora:
    - quarterly financial reports
    - company websites
    - google|yelp reviews
    - ...
- score companies:
    - S&P 500
    - SA Top 40
    - SA SMEs
    - SA Startups
    - YC Startups


## project structure:
- docs: project documentation lives in here.
- src: production code lives in folder and is divided in the modules below:
    - app_corporate_culture_measurer: project package
        - api:
            - the API to the application lives in this module.
            - the current implementation is a REST API, but a gRPC, CLI API, etc would be implemented in here.
        - config:
            - configurable values live in here. 
            - these are values such as Hand Ranks, Card Ranks.
                - as the system scales, you could migrate these into a database to allow independently
                changing config without restarting the application.
        - core:
            - the domain logic of the application lives in this module.
        - gateway:
            - all external interaction objects(e.g. files, external APIs etc) live in this module.
        - model:
            - the domain models for Poker live in this in this module.
        - app_corporate_culture_measurer.py:
            entry point to startup the application
- tests: test code lives in folder.
    the tests are intentionally separated from production code.
    - benefits:
        - tests can run against an installed version after executing `pip install .`.
        - tests can run against the local copy with an editable install after executing `pip install --edit`.
        - when using Docker, the entire app_corporate_culture_measurer folder can be copied without needing to exclude tests, which we don't release to PROD.
    - more in depth discussion here: https://docs.pytest.org/en/latest/explanation/goodpractices.html#choosing-a-test-layout-import-rules

- utilities: any useful scripts, such as curl & postman requests, JSON payloads, software installations, etc.

## setup instructions:
1. install `python 3.11` or lower.
    - [Python Download]: (https://www.python.org/downloads/)

2. clone repo: 
    ```shell
    git clone git@github.com:praisetompane/app_corporate_culture_measurer.git
    ```
3. Download and uncompress [Stanford CoreNLP v3.9.2](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip). Newer versions may work, but they are not tested. Either [set the environment variable to the location of the uncompressed folder](https://stanfordnlp.github.io/stanfordnlp/corenlp_client.html), or edit the following line in the `global_options.py` to the location of the uncompressed folder, for example: 
> os.environ["CORENLP_HOME"] = "/home/user/stanford-corenlp-full-2018-10-05/"   

- If you are using Windows, use "/" instead of "\\" to separate directories.  
- Make sure [requirements for CoreNLP](https://stanfordnlp.github.io/CoreNLP/) are met. For example, you need to have Java installed (if you are using Windows, install [Windows Offline (64-bit) version](https://java.com/en/download/manual.jsp)). To check if CoreNLP is set up correctly, use command line (terminal) to navigate to the project root folder and run `python -m culture.preprocess`. You should see parsed outputs from a single sentence printed after a moment:

    (['when[pos:WRB] I[pos:PRP] be[pos:VBD] a[pos:DT]....
## package management:
- install pipenv: https://pypi.org/project/pipenv/
- install packages into local environment using pipenv[**only required for first run**]:
    ```shell
    pipenv install
    ```
## run program:

- activate environment
    ```shell
    pipenv shell
    ```
- to start system run:
    ```shell
    ./start_system.sh
    ```

## testing:
### unit tests:
- to run tests:
    - activate environment
    ```shell
    pipenv shell
    ```
    - run tests
    ```shell
    pytest
    ```
        
### end to end tests:
- The curl request used can be found in `utilities/curl/`.
    - example:
    ```shell
   
    ```
    ![end to end curl example](./docs/end_to_end_curl_example.png) <br>

- If Postman requests can be found in `/utilities/postman/app_corporate_culture_measurer.postman_collection.json`.
    ![end to end postman example](./docs/end_to_end_postman_example.png)
    - Guide for how to import into Postman: https://learning.postman.com/docs/getting-started/importing-and-exporting/importing-data/


## development:
- to run system in debug mode:
    ```shell
    
    ```
- running in VSCode:
    - open the "Run and Debug" view:
    - click the green play button.
        - the server will inform you the host and port in the terminal output at the bottom.
        - from here you debug like normal(i.e. add break points, step into code definitions, evaluate code snippets, etc)
    ![start system output](./docs/vscode_debugging.png)
## git conventions:
- **NB:** the master is locked and all changes must come through a Pull Request.
- commit messages:
    - provide concise commit messages that describe what you have done.
        ```shell
        # example:
        git commit -m "feat(core): algorithm" -m"implement my new shiny faster algorithm"
        ```
    - screen shot of Githb view
    - references: 
        - https://www.conventionalcommits.org/en/v1.0.0/
        - https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/


**Disclaimer**: This is still work in progress.
======================================================================================================================================================
## Data
We included some example data in the `data/input/` folder. The three files are
- `documents.txt`: Each line is a document (e.g., each earnings call). Each document needs to have line breaks remvoed. The file has no header row. 
- `document_ids.txt`: Each line is document ID (e.g., unique identifier for each earnings call). A document ID cannot have `_` or whitespaces. The file has no header row. 
- (Optional) `id2firms.csv`: A csv file with three columns (`document_id`:str, `firm_id`:str, `time`:int). The file has a header row. 

## Before running the code
You can config global options in the `global_options.py`. The most important options are perhaps:
- The RAM allocated for CoreNLP
- The number of CPU cores for CoreNLP parsing and model training
- The seed words
- The max number of words to include in each dimension. Note that after filtering and deduplication (each word can only be loaded under a single dimension), the number of words will be smaller. 

## Running the code
1. Use `python parse.py` to use Stanford CoreNLP to parse the raw documents. This step is relatvely slow so multiple CPU cores is recommended. The parsed files are output in the `data/processed/parsed/` folder:
    - `documents.txt`: Each line is a *sentence*. 
    - `document_sent_ids.txt`: Each line is a id in the format of `docID_sentenceID` (e.g. doc0_0, doc0_1, ..., doc1_0, doc1_1, doc1_2, ...). Each line in the file corresponds to `documents.txt`. 
    
    Note about performance: This step is time-consuming (~10 min for 100 calls). Using `python parse_parallel.py` can speed up the process considerably (~2 min with 8 cores for 100 calls) but it is not well-tested on all platforms. To not break things, the two implementations are separated. 

2. Use `python clean_and_train.py` to clean, remove stopwords, and named entities in parsed `documents.txt`. The program then learns corpus specific phrases using gensim and concatenate them. Finally, the program trains the `word2vec` model. 

    The options can be configured in the `global_options.py` file. The program outputs the following 3 output files:
    - `data/processed/unigram/documents_cleaned.txt`: Each line is a *sentence*. NERs are replaced by tags. Stopwords, 1-letter words, punctuation marks, and pure numeric tokens are removed. MWEs and compound words are concatenated. 
    - `data/processed/bigram/documents_cleaned.txt`: Each line is a *sentence*. 2-word phrases are concatenated.  
    - `data/processed/trigram/documents_cleaned.txt`: Each line is a *sentence*. 3-word phrases are concatenated. This is the final corpus for training the word2vec model and scoring. 

   The program also saves the following gensim models:
   - `models/phrases/bigram.mod`: phrase model for 2-word phrases
   - `models/phrases/trigram.mod`: phrase model for 3-word phrases
   - `models/w2v/w2v.mod`: word2vec model
   
3. Use `python create_dict.py` to create the expanded dictionary. The program outputs the following files:
    - `outputs/dict/expanded_dict.csv`: A csv file with the number of columns equal to the number of dimensions in the dictionary (five in the paper). The row headers are the dimension names. 
    
    (Optional): It is possible to manually remove or add items to the `expanded_dict.csv` before scoring the documents. 

4. Use `python score.py` to score the documents. Note that the output scores for the documents are not adjusted by the document length. The program outputs three sets of scores: 
    - `outputs/scores/scores_TF.csv`: using raw term counts or term frequency (TF),
    - `outputs/scores/scores_TFIDF.csv`: using TF-IDF weights, 
    - `outputs/scores/scores_WFIDF.csv`: TF-IDF with Log normalization (WFIDF). 

    (Optional): It is possible to use additional weights on the words (see `score.score_tf_idf()` for detail).  

5. (Optional): Use `python aggregate_firms.py` to aggregate the scores to the firm-time level. The final scores are adjusted by the document lengths. 
