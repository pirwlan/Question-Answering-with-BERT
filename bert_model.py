import os
import utils


def prepare_model():
    utils.create_folder(os.getenv('MODEL_PATH'))
