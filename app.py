from flask import Flask, render_template, request
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

import data_processing as dp

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

    answer = dp.get_answer(decoded_ids, start_pos, stop_pos)

    return render_template('index.html', pred=f'What a tough question...\n is it {answer}?')


if __name__ == '__main__':

    bert_model = TFAutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

    app.run(host='0.0.0.0', port=8000, debug=True)