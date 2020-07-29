from dotenv import load_dotenv

import bert_model as bm

def main():
    model = bm.prepare_model()


if __name__ == '__main__':
    load_dotenv('config.env')
    main()
