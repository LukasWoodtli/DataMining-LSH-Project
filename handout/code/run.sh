cat ../data/training.txt | python mapper.py | sort | python reducer.py > pairs.txt
python check.py pairs.txt ../data/duplicates.txt
