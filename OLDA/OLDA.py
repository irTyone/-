import json
import argparse
from gensim import corpora
from gensim.models.ldamodel import LdaModel




def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def build_corpus(vocab_json: str, doc_freq_json: str):
    """
    vocab_tfidf.json : dict { "0": "考试", "1": "公务员", ... }
    freq_tfidf.json  : list[dict] [ { "0": 2, "5": 1 }, ... ]
    """

    vocab_raw = load_json(vocab_json)
    docs = load_json(doc_freq_json)

    # -------- 校验 vocab --------
    if isinstance(vocab_raw, list):
        raise TypeError(
            f"[ERROR] 你把 freq 文件当成 vocab 传进来了: {vocab_json}"
        )
    if not isinstance(vocab_raw, dict):
        raise TypeError(
            f"[ERROR] vocab_tfidf.json 必须是 dict {{id: word}}"
        )

    # -------- 校验 docs --------
    if not isinstance(docs, list):
        raise TypeError(
            f"[ERROR] freq_tfidf.json 必须是 list[dict]"
        )

    # -------- 构建 dictionary --------
    id2token = {int(i): w for i, w in vocab_raw.items()}

    dictionary = corpora.Dictionary()
    dictionary.id2token = id2token
    dictionary.token2id = {w: i for i, w in id2token.items()}
    dictionary.num_terms = len(id2token)

 
    corpus = []
    for doc in docs:
        bow = [(int(wid), int(freq)) for wid, freq in doc.items()]
        corpus.append(bow)

    return corpus, dictionary



def train_lda(
    corpus,
    dictionary,
    num_topics=50,
    passes=1,
    iterations=50,
    chunksize=256
):
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,         
        iterations=iterations,
        alpha="auto",
        eta="auto",
        chunksize=chunksize,
        random_state=42,
        eval_every=None
    )
    return lda_model




def extract_topics(lda_model, num_topics, top_words=15):
    topics = {}
    for i, topic in lda_model.show_topics(
        num_topics=num_topics,
        num_words=top_words,
        formatted=False
    ):
        topics[i] = [word for word, prob in topic]
    return topics


def get_document_topics(lda_model, corpus):
    """
    返回可 JSON 序列化的文档-主题分布
    关键修复点：numpy.float32 -> Python float
    """
    results = []
    for doc in corpus:
        topics = lda_model.get_document_topics(
            doc,
            minimum_probability=0.0
        )
        results.append([
            [int(tid), float(prob)]
            for tid, prob in topics
        ])
    return results




def main():
    parser = argparse.ArgumentParser(description="LDA / OLDA 训练脚本（gensim）")

    parser.add_argument("--vocab", required=True, help="vocab_tfidf.json")
    parser.add_argument("--docs", required=True, help="freq_tfidf.json")

    parser.add_argument("--num_topics", type=int, default=50)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--chunksize", type=int, default=256)

    parser.add_argument("--top_words", type=int, default=15)
    parser.add_argument("--topics_out", default="/home/liuyuan/Class_data/model/topics.json")
    parser.add_argument("--doc_topics_out", default="/home/liuyuan/Class_data/model/doc_topics.json")
    parser.add_argument("--model_out", default="lda.model")

    args = parser.parse_args()

    # ---- 构建语料 ----
    corpus, dictionary = build_corpus(args.vocab, args.docs)

    print(f"[INFO] 文档数: {len(corpus)}")
    print(f"[INFO] 词表大小: {dictionary.num_terms}")

    # ---- 训练 LDA / OLDA ----
    lda_model = train_lda(
        corpus=corpus,
        dictionary=dictionary,
        num_topics=args.num_topics,
        passes=args.passes,
        iterations=args.iterations,
        chunksize=args.chunksize
    )

    # ---- 保存模型 ----
    lda_model.save(args.model_out)

    # ---- 输出主题词 ----
    topics = extract_topics(
        lda_model,
        num_topics=args.num_topics,
        top_words=args.top_words
    )
    save_json(topics, args.topics_out)

    # ---- 输出文档主题分布 ----
    doc_topics = get_document_topics(lda_model, corpus)
    save_json(doc_topics, args.doc_topics_out)

    print(f"[OK] LDA 训练完成")
    print(f"[OK] 主题词文件: {args.topics_out}")
    print(f"[OK] 文档主题分布: {args.doc_topics_out}")
    print(f"[OK] 模型文件: {args.model_out}")


if __name__ == "__main__":
    main()
