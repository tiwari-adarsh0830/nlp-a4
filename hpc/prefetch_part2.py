# Run once on the LOGIN node inside the a4p2 env.
import nltk
from transformers import T5TokenizerFast, T5ForConditionalGeneration

nltk.download("punkt")
T5TokenizerFast.from_pretrained("google-t5/t5-small")
T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
print("part2 cache ready")
