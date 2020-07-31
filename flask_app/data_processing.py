from transformers import AutoTokenizer

import numpy as np
import os


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


def construct_answer(tokens):
    """
    Combine tokens into a string, remove some hash symbols, and leading/trailing whitespace.

    Args:
        tokens: list -  a list of tokens

    Returns:
        out_string: string - the processed string.
    """

    for idx, curr_token in enumerate(tokens):
        if idx == 0:
            out_string = curr_token

        elif idx > 0:

            if curr_token[0] == '#':

                curr_token = curr_token.replace("##", "")

            else:
                curr_token = " " + curr_token

            out_string += curr_token

    out_string = out_string.strip()

    return out_string


def get_answer(decoded_ids, answer_start, answer_stop):
    """

    Args:
        decoded_ids: list - list of decoded tokens
        answer_start: list - list of all likelyhood for start for each position
        answer_stop: list - list of all likelyhood for stop each position

    Returns:
        answer: string - most likely answer
    """

    start_pos_best = -1
    stop_pos_best = -1

    sum_max = -np.inf

    for curr_start_pos in range(len(answer_start)):

        for curr_stop_pos in range(curr_start_pos, len(answer_stop)):

            if answer_start[curr_start_pos] + answer_stop[curr_stop_pos] > sum_max:
                sum_max = answer_start[curr_start_pos] + answer_stop[curr_stop_pos]#

                start_pos_best = curr_start_pos

                stop_pos_best = curr_stop_pos

    answer = construct_answer(decoded_ids[start_pos_best:stop_pos_best + 1])

    return answer
