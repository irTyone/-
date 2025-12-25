import argparse
import json
import csv
from collections import Counter
from gensim.models import LdaModel
import jieba

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv(path, text_column="微博正文"):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_column, "").strip()
            if text:
                texts.append(text)
    return texts

def text_to_bow(text, vocab):
    """
    文本 -> {word_id: freq}，忽略词表中没有的词
    """
    words = jieba.lcut(text)
    word2id = {w: int(i) for i, w in vocab.items()}
    bow_counter = Counter()
    for w in words:
        if w in word2id:
            bow_counter[word2id[w]] += 1
    return dict(bow_counter)

def infer_texts(texts, lda_model_path, vocab_path, topics_json_path=None, top_k=5):
    lda = LdaModel.load(lda_model_path)
    vocab = load_json(vocab_path)
    topic_words = load_json(topics_json_path) if topics_json_path else None

    all_doc_topics = []
    for idx, text in enumerate(texts):
        bow = text_to_bow(text, vocab)
        if not bow:
            print(f"[WARN] 文档 {idx} 中没有训练词表里的词，跳过")
            all_doc_topics.append([])
            continue

        doc_topics = lda.get_document_topics(bow, minimum_probability=0.0)
        doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
        all_doc_topics.append(doc_topics)

        print(f"\n[INFO] 文档 {idx} 主题分布（Top {top_k}）:")
        for tid, prob in doc_topics[:top_k]:
            print(f"Topic {tid:>2d} | prob = {prob:.4f}")
            if topic_words and str(tid) in topic_words:
                words = topic_words[str(tid)]
                print("   keywords:", " ".join(words[:8]))

    return all_doc_topics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OLDA 文档主题推理（CSV/JSON 支持）")
    parser.add_argument("--model", required=True, help="已训练好的 lda.model")
    parser.add_argument("--vocab", required=True, help="vocab_tfidf.json")
    parser.add_argument("--topics", default=None, help="topics.json，可选，用于显示关键词")
    parser.add_argument("--input", required=True, help="输入文件路径（CSV 或 JSON）")
    parser.add_argument("--input_type", choices=["csv","json"], required=True, help="输入文件类型")
    parser.add_argument("--top_k", type=int, default=5, help="显示前 K 个主题")
    parser.add_argument("--output", default=None, help="可选：保存 doc_topics.json")
    parser.add_argument("--text_column", default="微博正文", help="CSV 文件中正文列名")

    args = parser.parse_args()

    # 读取文本
    if args.input_type == "csv":
        texts = load_csv(args.input, text_column=args.text_column)
    else:
        raw_json = load_json(args.input)
        texts = [doc.get(args.text_column, "").strip() for doc in raw_json if doc.get(args.text_column)]

    # 推理
    all_doc_topics = infer_texts(
        texts,
        lda_model_path=args.model,
        vocab_path=args.vocab,
        topics_json_path=args.topics,
        top_k=args.top_k
    )

    # 可选保存
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_doc_topics, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 文档主题分布已保存到 {args.output}")
