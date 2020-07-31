# Bert answers all your questions!

NLP is one of the hottest areas in ML/DL, particularly transformers, a kind of network architecture that has revolutionized the field. Here I used distilBERT, I variant of the Bert model. which is lighter and faster than Bert. Even though the model size has been reduced by 40%, the performance was only reduced by 3% [1](https://arxiv.org/abs/1910.01108). The small size makes distilBERT perfect for edge devices. 

I used distilBERT, particularly a model provided by [Hugging Face](https://huggingface.co/) as a question-answering tool. In a simple web app You can enter a phrase or a paragraph, and a question. The model will then calculate the most likely answer, and this answer will be shown in a human friendly format. 



![webapp](img/img.png 'webapp')





## How to use it

You want to see it for yourself? Quiet easy, you only need Docker. 

### Step 1: Clone the repo:

 ```bash
git clone git@github.com:pirwlan/QA_Bert.git
 ```



### Step 2 (optional): Download the model

Then, [download the model](https://cdn.huggingface.co/distilbert-base-cased-distilled-squad-tf_model.h5), rename it to `tf_model.h5` and save it in the model folder in the repository. If you do this, before you build the image, the model is downloaded only once. If you decide not to download it now, it has to be downloaded every time the image is run, which can take a bit of time, dependent on your internet connection. 



### Step 3: Make Docker image

Go to the QA_Bert folder, and enter:

```bash
 sudo docker build -t bert_in_da_box  -f ./flask_app/Dockerfile .
```

This downloads all the necessary base images, modules and so on, and bundles everything in one single image.



### Step 4: Run Docker image

```bash
sudo docker run -p 8000:8000 bert_in_da_box
```

Run this, go to your web browser, and go to http://0.0.0.0:8000/

There you can enter a question and a paragraph containing the answer. Once you press predict, you should see the (hopefully) correct answer within a few seconds.

## 
