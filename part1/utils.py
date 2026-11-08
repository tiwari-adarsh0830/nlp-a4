import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# Neighbouring keys on a QWERTY keyboard
QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh", "u": "yihj",
    "i": "uojk", "o": "ipkl", "p": "ol", "a": "qwsz", "s": "awedxz", "d": "serfcx",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}

SYNONYM_PROB = 0.3  # probability of replacing an eligible word with a synonym
TYPO_PROB = 0.1     # probability of introducing a keyboard typo into an eligible word


def _match_case(src, tgt):
    if src.isupper():
        return tgt.upper()
    if src[0].isupper():
        return tgt[0].upper() + tgt[1:]
    return tgt


def _get_synonym(word):
    candidates = set()
    for syn in wordnet.synsets(word.lower()):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower() and name.isascii():
                candidates.add(name)
    if not candidates:
        return None
    return _match_case(word, random.choice(sorted(candidates)))


def _add_typo(word):
    positions = [i for i, ch in enumerate(word) if ch.lower() in QWERTY_NEIGHBORS]
    if not positions:
        return word
    i = random.choice(positions)
    ch = word[i]
    new_ch = random.choice(QWERTY_NEIGHBORS[ch.lower()])
    if ch.isupper():
        new_ch = new_ch.upper()
    return word[:i] + new_ch + word[i + 1:]


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # Transformation: synonym replacement (WordNet) + QWERTY keyboard typos.
    # Only alphabetic words longer than 3 characters are touched, so short function
    # words and negations like "not"/"no" stay intact and the label is preserved.
    tokens = word_tokenize(example["text"])
    new_tokens = []
    for tok in tokens:
        if tok.isalpha() and len(tok) > 3:
            if random.random() < SYNONYM_PROB:
                syn = _get_synonym(tok)
                if syn is not None:
                    tok = syn
            if random.random() < TYPO_PROB:
                tok = _add_typo(tok)
        new_tokens.append(tok)

    example["text"] = TreebankWordDetokenizer().detokenize(new_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
