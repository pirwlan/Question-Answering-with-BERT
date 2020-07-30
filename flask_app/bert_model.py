from transformers import TFAutoModelForQuestionAnswering

import os
import requests
from flask_app import utils


def download_bert():
    """
    Downloads Bert model for QA if not already downloaded
    """
    url = os.getenv('BERT_URL')
    model_path = os.path.join(os.getenv('BERT'))

    r = requests.get(url, allow_redirects=True)
    open(model_path, 'wb').write(r.content)


def prepare_model():
    utils.create_folder(os.getenv('MODEL_PATH'))
    if not os.path.isfile(os.getenv('BERT')):

        download_bert()

    bert_model = TFAutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
    # bert_model.save_weights('BertQA')
    return bert_model
