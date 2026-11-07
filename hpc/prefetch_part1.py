# Run once on the LOGIN node (has internet) inside the a4p1 env.
# Caches everything Part 1 downloads at runtime, in case compute nodes are offline.
import nltk, evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

for pkg in ["wordnet", "omw-1.4", "punkt"]:
    nltk.download(pkg)  # goes to $NLTK_DATA set in /ext3/env.sh
load_dataset("imdb")
AutoTokenizer.from_pretrained("bert-base-cased")
AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
evaluate.load("accuracy")  # evaluate==0.4.0 fetches this metric script from the Hub
print("part1 cache ready")
