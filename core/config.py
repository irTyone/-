import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# vocab 文件
VOCAB_PATH = os.path.join(BASE_DIR, "..", "vocab", "vocab.json")


CSV_DIR = '/home/xironghui/homework/spider_output'

STOP_LIST = os.path.join(BASE_DIR, "..", "vocab", "stopwords.txt")


CONTENT_INFO = os.path.join(BASE_DIR, "..", "content_info", "content.json")

DATA_PATH = os.path.join(BASE_DIR, "..", "data")
