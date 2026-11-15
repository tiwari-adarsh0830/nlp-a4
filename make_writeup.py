"""Build the written-answers PDF for the Gradescope writeup slot.

Anything still missing is written as [[TODO: ...]] and shows up in red in the PDF,
so search this file for TODO before submitting.

Run:  python make_writeup.py  ->  A4_writeup.pdf
"""

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (KeepTogether, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "A4_writeup.pdf"
STUDENT = "Adarsh Tiwari (at7095)"


# ---------------------------------------------------------------------------
# Content. Each section is (heading, question prompt or None, markdown-ish body).
# Body format: blank line = new paragraph, lines starting with "- " = bullets,
# lines starting with "|" = table (first row is the header), `code`, **bold**.
# ---------------------------------------------------------------------------

TOP = """
**GitHub repo:** github.com/tiwari-adarsh0830/nlp-a4

**Q7 checkpoint (Google Drive):** https://drive.google.com/drive/folders/1bS8lZic_Odo5EUVBgQHK5arLovloeYBp

**Discussed with:** Sheetal

**AI use:** I used Claude Code as a coding assistant to write and debug much of the code in both parts and to help draft this writeup. I ran all the experiments on Torch myself and checked the code, results and writeup.
"""

PART1 = [
    ("Q1. Fine-tuning BERT on IMDB", None, """
The training loop is in `do_train` in main.py. For every epoch and every batch it moves the batch to the GPU, runs the model, takes `outputs.loss`, calls backward, steps the optimizer and the linear lr scheduler, and zeroes the gradients. I used the default settings (bert-base-cased, 3 epochs, lr 5e-5, batch size 8).

Accuracy on the full IMDB test set (25,000 reviews): **93.12%** (out_original.txt).
"""),

    ("Q2(a). The transformation",
     "Design a transformation of the dataset, and explain the details of what it does. You should "
     "describe it clearly, such that someone who reads the description can reimplement it. You should "
     "also include why that is a “reasonable” transformation, i.e., something which could potentially "
     "be an input at test time from a user. If you like, you can use LLMs to perform the data "
     "transformation.", """
I combined two things people actually do when they write reviews: using a different word for the same thing, and hitting the wrong key.

The review is split into words with NLTK's `word_tokenize` and joined back with `TreebankWordDetokenizer`. A word is only eligible if it is all letters and longer than 3 characters. That leaves punctuation, numbers and short words alone, including "not", "no" and "bad", so the sentiment can't get flipped by accident. Every eligible word then goes through two independent steps:

- **Synonym swap (p = 0.3).** Look the word up in WordNet and keep only its most common sense, which I take to be the synset where this exact word has the highest SemCor count (`lemma.count()` > 0; ties go to WordNet's first listed sense). Pick a random single-word lemma from that synset that isn't the word itself and copy the original capitalization. If no sense or lemma qualifies, the word stays the same.
- **Typo (p = 0.35).** Pick one random letter in the word and replace it with a key next to it on a QWERTY keyboard, keeping the case (friends -> friehds).

A word can get both. My first version took synonyms from every sense of the word, which gave swaps like watch -> ticker and boring -> drilling. Using only the most common sense fixed most of those, and requiring an exact match on the word form stopped swaps like kept -> maintain that lose the tense.

It's reasonable because both changes happen in real reviews all the time. People say film instead of movie, dull or tedious instead of boring, amusing instead of funny, and typing fast on a keyboard or phone gives exactly this kind of one-letter slip. A person reading the transformed review still knows right away if it's positive or negative. A few swaps are still a bit odd (time -> clip, simply -> but), because WordNet really does list them as synonyms of the main sense, but they don't change what the review is saying.

Example: a positive review from the IMDB test set, run through my final transformation (p = 0.3 synonyms, p = 0.35 typos).

**Before:** I was truly and wonderfully surprised at "O' Brother, Where Art Thou?" The video store was out of all the movies I was planning on renting, so then I came across this. I came home and as I watched I became engrossed and found myself laughing out loud.

**After:** I was genuineoy and wonderfully surprised at "O' Brother, Where Art Tnou?" The video shop was out of all the movies I was planning on renting, so then I czme across this . I cane home and as I watched I became engrossed and found mhself laughung out loud.

Two words got synonyms (store -> shop, truly -> genuinely) and six got a one-letter typo, including genuinely right after its synonym swap. The "this ." spacing comes from re-joining the tokens. The review is still obviously positive.
"""),

    ("Q2(b). Accuracy on the transformed test set", None, """
The same model from Q1 gets 87.86% on the transformed test set vs 93.12% on the original test set, a drop of 5.26 points (out_transformed.txt).
"""),
]

PART1_Q3 = ("Q3. Data augmentation",
    "Evaluate the performance of the model trained above. Your objective is to assess how data "
    "augmentation affects the model’s ability to handle both original and transformed test data. "
    "Report the accuracy values for both the original and transformed test data evaluations. Analyze "
    "and discuss: (1) Did the model’s performance on the transformed test data improve after applying "
    "data augmentation? (2) How did data augmentation affect the model’s performance on the original "
    "test data? Did it enhance or diminish its accuracy? Offer an intuitive explanation for the observed "
    "results, considering the impact of data augmentation on model training. Explain one limitation of "
    "the data augmentation approach used here to improve performance on out-of-distribution (OOD) test "
    "sets.", """
I took 5,000 random training reviews (seed 42), ran them through the transformation from Q2(a) and added them to the full 25,000-review training set, so 30,000 examples in total. Then I trained a new model with the same settings as Q1.

| Model | Original test | Transformed test |
| Original training data (Q1, Q2) | 93.12% | 87.86% |
| Augmented training data | 92.89% | 89.19% |

(1) Yes. Transformed-test accuracy went from 87.86% to 89.19%, so augmentation helped by 1.33 points. It didn't close the whole gap though: the augmented model is still about 3.7 points worse on transformed text than on clean text.

(2) Original-test accuracy went from 93.12% to 92.89%, so it went down by 0.24 points. That's small enough to be within the noise of a single training run, so I'd call it basically unchanged.

The augmented model has seen 5,000 reviews with synonyms and typos that keep the same label as the clean ones, so it learns that friehds and friends, or boring and tedious, point the same way. The original model only ever saw clean text. When a typo hits a word, BERT's WordPiece splits it into unusual subword pieces that no longer carry the original word's meaning, so the model loses that word's signal. The original-test number barely moved because the model still saw all 25,000 clean reviews. The 5,000 extra examples are noisy copies of reviews it already had, so they don't change much about how it handles clean text.

Limitation: it only helps with the shift I trained on. The 5,000 extra reviews only have my two kinds of noise, so the model gets better at exactly those and not at other out-of-distribution changes, like reviews from a different domain, slang, sarcasm, or typos that aren't keyboard neighbors. It also doesn't add any new content, since the transformed reviews are copies of reviews that are already in the training set.
""")

PART2 = [
    ("Q4. Data statistics",
     "Describe the data statistics before and after any pre-processing by filling in Table 1 and "
     "Table 2. Use the T5 tokenizer to report the statistics. Depending on your pre-processing, some "
     "numbers may be identical across tables.", """
Everything is measured with the google-t5/t5-small tokenizer. Lengths are in tokens, and vocabulary size is the number of different token ids that show up. "Before" is the raw text with no `</s>`. "After" is exactly what the model sees: `</s>` added at the end and `<` swapped for `LT` in the SQL (see Q5).

**Table 1. Before pre-processing**

| Statistic | Train | Dev |
| Number of examples | 4,225 | 466 |
| Mean sentence length | 17.10 | 17.07 |
| Mean SQL query length | 216.37 | 210.05 |
| Vocabulary size (natural language) | 791 | 465 |
| Vocabulary size (SQL) | 555 | 395 |

**Table 2. After pre-processing**

| Statistic | Train | Dev |
| Model name | T5 fine-tuned (google-t5/t5-small) | T5 fine-tuned (google-t5/t5-small) |
| Mean sentence length | 18.10 | 18.07 |
| Mean SQL query length | 217.37 | 211.05 |
| Vocabulary size (natural language) | 792 | 466 |
| Vocabulary size (SQL) | 556 | 396 |

The lengths go up by exactly 1 because of the `</s>`. The more important change doesn't show up in the means: before pre-processing, 393 train and 45 dev SQL queries contained an `<unk>` token, and after it none do. The longest SQL query is 511 tokens (train) and 503 (dev). The test set has 432 questions.
"""),

    ("Q5. T5 fine-tuning details",
     "Use Table 3 to describe your data processing steps (if any) and implementation details for the "
     "fine-tuned T5 model. Be clear enough that one can replicate your approach in PyTorch using only "
     "your descriptions.", """
**Table 3. Best T5 fine-tuning setup**

| Design choice | Description |
| Data processing | The questions are used as they are. In the SQL, I replace every `<` with the word `LT` before tokenizing and turn it back into `<` after decoding. T5's vocabulary has no `<`, so without this it becomes `<unk>` and disappears when decoded. That breaks about 10% of the gold queries (anything like `arrival_time < 1200`), and the model could never get them exactly right. `LT` never appears anywhere in the data, and every train and dev query comes back exactly the same after encoding and decoding. |
| Tokenization | Both the encoder and the decoder use the google-t5/t5-small tokenizer, which adds `</s>` at the end. The encoder input is the question. The decoder input is `<extra_id_0>` (used as the start token everywhere) followed by the SQL tokens without the last one, and the target is the full SQL ending in `</s>`. Each batch is padded with 0 to its longest example, with an attention mask for the encoder, and padded target positions are left out of the loss. At evaluation I use greedy decoding from the same `<extra_id_0>` start token, with at most 600 new tokens, then decode with special tokens skipped and swap `LT` back to `<`. |
| Architecture | Full fine-tuning of the pretrained T5-small. Nothing is frozen: the shared embeddings, all encoder and decoder layers and the LM head (tied to the embeddings) are all trained, 60.5M parameters in total. |
| Hyperparameters | AdamW with learning rate 1e-3, weight decay 0, batch size 16, cosine schedule with 1 warmup epoch, up to 20 epochs, early stopping on dev record F1 with patience 5 epochs, and the checkpoint with the best dev F1 is kept. |

I didn't train a version without the `LT` swap, so I'm not claiming it raises the score. It's needed so that those queries can be generated at all.
"""),

    ("Q6. Results and error analysis",
     "Quantitative results: Use Table 4 to report your test and development set results. Your test "
     "results should match the results on the leaderboard. For the development set, also report results "
     "from experiments you conducted to arrive at your final configuration. Qualitative error analysis: "
     "Conduct a detailed error analysis for your model. Identify common error types and discuss possible "
     "reasons for these errors in Table 5. You must identify at least three classes of errors for the "
     "queries, and use examples to illustrate them.", """
**Table 4. Results**

| Run | Split | Query EM | Record F1 |
| lr 1e-3, 1 epoch | Dev | 17.17 | 32.88 |
| lr 1e-3, up to 20 epochs (final) | Dev | 73.61 | 88.02 |
| lr 3e-4, up to 20 epochs | Dev | 61.80 | 79.90 |
| Final model (lr 1e-3) | Test | 0.00 | 81.79 |

Record EM for the final model is 85.62 on dev (399/466) and 79.40 on test.

lr 1e-3 did best, with 88.02 dev F1 vs 79.90 for lr 3e-4 (about 8 points higher), so it's my final model. lr 3e-4 learned more slowly right from the start: its dev F1 after the first three epochs was 0.16 / 0.37 / 0.49, vs 0.33 / 0.55 / 0.65 for lr 1e-3. It also plateaued lower. From epoch 12 on it stayed between 0.77 and 0.80, its best dev F1 (0.799) was at epoch 14, and after 5 more epochs with no improvement, early stopping (patience 5) ended it after epoch 19, one short of the 20-epoch budget. So more epochs probably wouldn't have closed the gap. One epoch (32.88 F1) was nowhere near enough training: the model hadn't learned the long, very regular SQL patterns yet.

**Table 5. Error analysis on the dev set (best run, lr 1e-3)**

| Error type | Example | Description | Statistics |
| SQL execution error | #138 "flights on continental leaving seattle on sunday after 430pm": the prediction matches the gold query exactly but ends with `... > 1630 ) ) )`, one `)` too many | The query doesn't run. 20 have unbalanced parentheses (`near ")"` or `incomplete input`), 8 use a table alias that isn't in FROM or an ambiguous column, and 1 timed out. 11 of these still scored F1 = 1.0 because the gold query returns no rows either. Likely reason: gold queries nest up to 8 levels of parentheses and can end with 7 `)` in a row, and the model has no explicit way to count them, so being off by one breaks the query. The FROM list is also written before WHERE, so the model has to commit to its tables before it knows which ones the conditions will use. | 29/466 |
| Wrong structure or SELECT target | #20 "give me the flight times ... from boston to baltimore in the morning before 8": gold `SELECT DISTINCT flight_1.departure_time`, predicted `SELECT DISTINCT flight_1.departure_time , flight_1.arrival_time` | The filters are fine but the query asks for the wrong thing: an extra or missing SELECT column, the wrong table (airline instead of airport for "what does iah mean"), or a wrong join. Likely reason: some questions are ambiguous ("flight times" could mean departure or arrival), and 73% of training queries are plain flight searches (`SELECT DISTINCT flight_1.flight_id`), so the model leans toward that pattern for less common question types. | 19/466 |
| Wrong constant | #70 "nonstop flights from las vegas to new york on america west": gold `airline_code = 'HP'`, predicted `airline_code = 'AA'` | Same structure as the gold query but one value is wrong: an airline code, a date or a time. Time windows for "around N o'clock" are often off by an hour or collapse to one point (`>= 2130 AND <= 2130`). Likely reason: rare names lose to common ones. "america west" is in only 1 training question while "american" is in 135, and `HP` shows up in 8 training queries vs 136 for `AA`. Times like "around 4 o'clock" also need a window around the hour, which is arithmetic the model can only copy from patterns. | 16/466 |
| Missing or extra condition | #427 "is there a continental flight leaving from las vegas to new york nonstop": the prediction drops `flight_1.airline_code = 'CO' AND(` | A condition from the question gets dropped (airline, time of day, the return leg of a round trip) or one gets added that wasn't asked for (an extra destination). Likely reason: the SQL targets average about 216 tokens and most of that is the same city/airport join pattern, so a condition mentioned once in the question is easy to lose, and an extra one gets added when the question looks like the common "from X to Y" pattern. | 12/466 |
| Gold label is wrong | #216 "flights from atlanta to washington dc leaving on thursday before 0900": the gold query is `SELECT state_1.state_code FROM state state_1 WHERE state_1.state_code = 'ASD';` | The gold query has nothing to do with the question, so a reasonable prediction still scores 0. Same for #340. #3 scores 1.0 only because its failed prediction and the bogus gold query both return no rows. Likely reason: annotation noise in the dataset (these gold queries look like a placeholder), not a model problem. | 3/466 |

These rows cover all 78 dev examples my model got wrong: the 67 whose records don't match the gold records (F1 below 1), plus the 11 SQL errors that still scored 1.0 because both queries return no rows. Each example has one type, except #3, which has a wrong gold query and also failed to run, so it's counted in both the first and the last row (29 + 19 + 16 + 12 + 3 = 79).
"""),

    ("Q7. Test performance", None, """
My test predictions are in submission/t5_ft_experiment_test.sql and submission/t5_ft_experiment_test.pkl in my GitHub repo (the same files I uploaded to Gradescope). On Gradescope my test set results are record F1 81.79 and record EM 79.40, and 37 of the 432 test queries raised SQL errors. Gradescope reports SQL query EM as 0.00 on the test set. The TA confirmed this number is computed and noted that exact match is very strict, since the query string has to match the gold one exactly, including spaces and commas. So even with good record F1 the test query EM can be low. On dev, where I compare against the provided dev.sql, my query EM was 73.61.
"""),
]


# ---------------------------------------------------------------------------
# Rendering (same look as the A3 writeup)
# ---------------------------------------------------------------------------

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(s):
    s = esc(s)
    s = re.sub(r"`([^`]+)`", lambda m: "<font name='Courier'>%s</font>" % m.group(1).replace(" ", "&nbsp;"), s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\[\[(TODO[^\]]*)\]\]", r"<font color='#c0392b'><b>[\1]</b></font>", s)
    s = s.replace(" -&gt; ", " &#8594; ")
    return s


def markdown_to_flowables(md, body, small):
    out = []
    for block in re.split(r"\n\s*\n", md.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        if all(l.strip().startswith("|") for l in lines):
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
            data = [[Paragraph(inline(c), small) for c in r] for r in rows]
            ncol = len(rows[0])
            if ncol == 2:
                widths = [110, 386]
            elif rows[0][0] == "Error type":
                widths = [70, 160, 210, 56]
            else:
                widths = None
            t = Table(data, hAlign="LEFT", repeatRows=1, colWidths=widths)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c4ccd6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            out += [t, Spacer(1, 6)]
        elif all(l.lstrip().startswith("- ") for l in lines):
            items = [ListItem(Paragraph(inline(l.lstrip()[2:]), body), leftIndent=12) for l in lines]
            out.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=12))
            out.append(Spacer(1, 4))
        else:
            out.append(Paragraph(inline(" ".join(l.strip() for l in lines)), body))
    return out


def build():
    styles = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=15,
                           leading=20, textColor=colors.HexColor("#1f2d40"), spaceAfter=4)
    sub = ParagraphStyle("S", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5,
                         leading=13, textColor=colors.HexColor("#5b6573"), alignment=1, spaceAfter=10)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=13,
                        leading=17, textColor=colors.HexColor("#1f2d40"), spaceBefore=10, spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
                        leading=15, textColor=colors.HexColor("#1d3a56"), spaceBefore=9, spaceAfter=3)
    q = ParagraphStyle("Q", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=9,
                       leading=12.5, textColor=colors.HexColor("#4a5563"), spaceAfter=5)
    body = ParagraphStyle("B", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.8,
                          leading=13.6, spaceAfter=6, alignment=TA_LEFT)
    small = ParagraphStyle("Sm", parent=body, fontSize=8.6, leading=11, spaceAfter=0)

    s = [Paragraph("DS-GA 1011 Assignment 4: Written Answers", title), Paragraph(STUDENT, sub)]
    s += markdown_to_flowables(TOP, body, small)

    def section(heading, prompt, text):
        head = [Paragraph(heading, h2)]
        if prompt:
            head.append(Paragraph(esc(prompt), q))
        s.append(KeepTogether(head))
        s.extend(markdown_to_flowables(text, body, small))

    s.append(Paragraph("Part 1", h1))
    for sec in PART1 + [PART1_Q3]:
        section(*sec)

    s.append(Paragraph("Part 2", h1))
    s.append(Paragraph(
        "All Part 2 runs were done on NYU Torch, fine-tuning google-t5/t5-small with the code in part2/.", body))
    for sec in PART2:
        section(*sec)

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=(612, 792), rightMargin=58, leftMargin=58,
                            topMargin=50, bottomMargin=50, title="DS-GA 1011 A4 Written Answers",
                            author=STUDENT)

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6b7685"))
        canvas.drawString(58, 28, "DS-GA 1011 A4 | " + STUDENT)
        canvas.drawRightString(554, 28, "Page %d" % d.page)
        canvas.restoreState()

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    todos = len(re.findall(r"\[\[TODO", Path(__file__).read_text())) - 1  # minus the one in the docstring
    print(f"Created: {OUTPUT}  ({todos} TODOs left)")


if __name__ == "__main__":
    build()
