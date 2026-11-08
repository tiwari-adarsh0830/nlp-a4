import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
T5_CHECKPOINT = 'google-t5/t5-small'
# The decoder's beginning-of-sequence token: one T5 sentinel, used everywhere
# (training decoder inputs, initial decoder inputs, and generate()).
START_TOKEN = '<extra_id_0>'

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = T5TokenizerFast.from_pretrained(T5_CHECKPOINT)
    return _tokenizer

START_TOKEN_ID = get_tokenizer().convert_tokens_to_ids(START_TOKEN)

# '<' is not in the T5 vocabulary: it tokenizes to <unk> and vanishes on decoding,
# which breaks ~10% of gold queries ("arrival_time < 1200"). Swap it for a
# placeholder word ("LT", never used in the data) and swap back after decoding.
LT_PLACEHOLDER = 'LT'

def encode_sql(query):
    '''SQL text -> text given to the tokenizer.'''
    return query.replace('<', LT_PLACEHOLDER)

def decode_sql(text):
    '''Inverse of encode_sql, applied to decoded model output.'''
    return re.sub(rf'\b{LT_PLACEHOLDER}\b', '<', text)

def decode_queries(token_ids):
    '''Model output ids (B x T) -> list of SQL strings.'''
    texts = get_tokenizer().batch_decode(token_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
    return [decode_sql(t).strip() for t in texts]

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = get_tokenizer()
        self.encoder_ids, self.decoder_ids = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_lines = load_lines(os.path.join(data_folder, f'{split}.nl'))
        # The tokenizer appends </s> to every sequence
        encoder_ids = [torch.tensor(ids, dtype=torch.long)
                       for ids in tokenizer(nl_lines)['input_ids']]

        if split == 'test':
            return encoder_ids, None

        sql_lines = load_lines(os.path.join(data_folder, f'{split}.sql'))
        assert len(sql_lines) == len(nl_lines)
        sql_lines = [encode_sql(q) for q in sql_lines]
        decoder_ids = [torch.tensor(ids, dtype=torch.long)
                       for ids in tokenizer(sql_lines)['input_ids']]
        return encoder_ids, decoder_ids

    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        if self.split == 'test':
            return self.encoder_ids[idx]
        return self.encoder_ids[idx], self.decoder_ids[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    enc_list = [enc for enc, _ in batch]
    sql_list = [sql for _, sql in batch]

    encoder_ids, encoder_mask, initial_decoder_inputs = _encode_batch(enc_list)

    start = torch.tensor([START_TOKEN_ID], dtype=torch.long)
    # Decoder input = <start> + SQL minus its last token; target = full SQL (ends in </s>)
    decoder_inputs = pad_sequence([torch.cat([start, sql[:-1]]) for sql in sql_list],
                                  batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(sql_list, batch_first=True, padding_value=PAD_IDX)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    return _encode_batch(batch)

def _encode_batch(enc_list):
    encoder_ids = pad_sequence(enc_list, batch_first=True, padding_value=PAD_IDX)
    # 1 for real tokens, 0 for padding (built from lengths, not from token values)
    encoder_mask = pad_sequence([torch.ones(len(enc), dtype=torch.long) for enc in enc_list],
                                batch_first=True, padding_value=0)
    initial_decoder_inputs = torch.full((len(enc_list), 1), START_TOKEN_ID, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    pass