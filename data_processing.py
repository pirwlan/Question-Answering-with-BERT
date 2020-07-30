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


def data_preprocessing(context, question, tokenizer):
    """
    Prepares input to Bert input format

    Args:
        context: str - Context where the answer is "hidden"
        question: str - Posed Question
        tokenizer: Bert.Tokenizer

    Returns:
        x_data: list - [input_ids, input_type_ids]
    """
    input_ids = tokenizer.encode(question, context)
    input_type_ids = make_input_type_ids(input_ids, tokenizer)

    x_data = [input_ids, input_type_ids]

    return x_data, input_ids