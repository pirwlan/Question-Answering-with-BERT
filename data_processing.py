from transformers import AutoTokenizer

import os


def tokenize(sentence):
    """
    Uses dist-bert tokeniezr to tokenize imput
    Args:
        sentence: string - input sequence

    Returns:
        tokenized input
    """
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
    return tokenizer.encode(sentence)


def make_input_ids(context, question):
    """

    Args:
        context:
        question:

    Returns:
        input_ids:
    """
    input_ids = context + question[1:]
    QA_type = [0] * len(context) + \
                     [1] * len(question[1:])

    return input_ids, QA_type


def padding(input_ids, QA_type, attention_mask):

    padding_length = int(os.getenv('MAX_LENGTH')) - len(input_ids)

    for element in [input_ids, QA_type, attention_mask]:
        element += ([0] * padding_length)

    return input_ids, QA_type, attention_mask


def data_preprocessing(context, question):
    context = tokenize(context)
    question = tokenize(question)

    input_ids, QA_type = make_input_ids(context, question)

    attention_mask = [1] * len(input_ids)

    input_ids, QA_type, attention_mask = padding(input_ids, QA_type, attention_mask)

    return input_ids, QA_type, attention_mask

