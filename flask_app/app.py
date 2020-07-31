from flask import Flask, render_template, request
from tensorflow import keras
from transformers import TFAutoModelForQuestionAnswering

import data_processing as dp
import tensorflow as tf

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html', pred='Please ask a question!')


@app.route('/predict', methods=['POST'])
def predict():
    data = [request.form['question'],
            request.form['context']]

    bert_input, decoded_ids = dp.data_preprocessing(context=data[1],
                                                    question=data[0])

    start_pos, stop_pos = bert_model.predict(bert_input)

    answer = dp.get_answer(decoded_ids, start_pos[0], stop_pos[0])

    return render_template('index.html', pred=f'What a tough question...\n I think the answer is {answer}?')


if __name__ == '__main__':

    bert_model = TFAutoModelForQuestionAnswering.from_pretrained('./model')
    app.run(host='0.0.0.0', port=8000, debug=True)
