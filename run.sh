echo "parse"
    python parse.py
echo "done"

echo "clean and train"
    python clean_and_train.py
echo "done"

echo "create culture dictionary"
    python create_dict.py
echo "done"

echo "score"
    python score.py
echo "done"

echo "aggregate firms"
    python aggregate_firms.py
echo "done"