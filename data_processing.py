from transformers import AutoTokenizer

import numpy as np


def make_input_type_ids(input_ids, tokenizer):
    """
    Makes mask to separate question and context

    Returns:
        input_type_ids: list - 0 for question token, 1 for context token
    """

    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)
    question_index = sep_index + 1

    context_index = len(input_ids) - question_index

    segment_ids = [0] * question_index + [1] * context_index

    assert len(segment_ids) == len(input_ids)

    return segment_ids


def data_preprocessing(context, question):
    """
    Prepares input to Bert input format

    Args:
        context: str - Context where the answer is "hidden"
        question: str - Posed Question

    Returns:
        x_data: list - [input_ids, input_type_ids]
    """

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')

    input_ids = tokenizer.encode(question, context)
    input_type_ids = make_input_type_ids(input_ids, tokenizer)

    x_data = [input_ids, input_type_ids]

    return x_data, tokenizer.convert_ids_to_tokens(input_ids)


def get_answer(decoded_ids, answer_start, answer_stop):
    """

    Args:
        decoded_ids: list - list of decoded tokens
        answer_start: int - start position of mostlikely start position
        answer_stop: int - stop position of most likely stop position

    Returns:
        answer: string - most likely answer
    """
    answer = decoded_ids[np.argmax(answer_start):np.argmax(answer_stop)][0]
    return answer
