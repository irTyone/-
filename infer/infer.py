import argparse
import json
import csv
from collections import Counter
from gensim.models import LdaModel
import jieba



# =========================================================
# jieba 缓存目录（避免 /tmp 写权限问题）
# =========================================================
jieba.dt.tmp_dir = "/home/liuyuan/.jieba_cache"


# =========================================================
# IO 工具
# =========================================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path, text_column="微博正文"):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_column, "")
            if text:
                text = text.strip()
                if text:
                    texts.append(text)
    return texts


# =========================================================
# 文本 -> gensim LDA bow
# =========================================================
def text_to_bow(text, vocab, lowercase=True):
    """
    text -> [(token_id, freq), ...]
    vocab: {"0": "中国", "1": "视频", ...}
    """
    words = jieba.lcut(text)

    if lowercase:
        words = [w.lower() for w in words]

    # vocab: id -> word  ==>  word -> id
    word2id = {w: int(i) for i, w in vocab.items()}

    bow_counter = Counter()
    for w in words:
        if w in word2id:
            bow_counter[word2id[w]] += 1

    return list(bow_counter.items())


# =========================================================
# 推理主逻辑
# =========================================================
def infer_texts(
    texts,
    lda_model_path,
    vocab_path,
    topics_json_path=None,
    top_k=5,
):
    print("[INFO] Loading LDA model...")
    lda = LdaModel.load(lda_model_path)

    print("[INFO] Loading vocab...")
    vocab = load_json(vocab_path)

    topic_words = load_json(topics_json_path) if topics_json_path else None

    all_doc_topics = []

    for idx, text in enumerate(texts):
        bow = text_to_bow(text, vocab)

        if not bow:
            print(f"[WARN] 文档 {idx} 无有效词（不在训练词表中）")
            all_doc_topics.append([])
            continue

        # LDA 推理
        doc_topics = lda.get_document_topics(
            bow, minimum_probability=0.0
        )

        # 按概率排序
        doc_topics = sorted(
            doc_topics, key=lambda x: x[1], reverse=True
        )

        # ======================
        # 打印（numpy.float32 可直接打印）
        # ======================
        print(f"\n[INFO] 文档 {idx} 主题分布（Top {top_k}）")
        for tid, prob in doc_topics[:top_k]:
            print(f"  Topic {tid:>2d} | prob = {prob:.4f}")
            if topic_words and str(tid) in topic_words:
                kws = topic_words[str(tid)]
                print("    keywords:", " ".join(kws[:8]))

        # ======================
        # 保存用（转成 JSON 可序列化）
        # ======================
        doc_topics_json = [
            [int(tid), float(prob)]
            for tid, prob in doc_topics
        ]

        all_doc_topics.append(doc_topics_json)

    return all_doc_topics


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LDA 文档主题推理（CSV / JSON）"
    )

    parser.add_argument(
        "--model", required=True, help="已训练好的 lda.model"
    )
    parser.add_argument(
        "--vocab", required=True, help="vocab.json（id -> word）"
    )
    parser.add_argument(
        "--topics", default=None, help="topics.json（可选）"
    )
    parser.add_argument(
        "--input", required=True, help="输入 CSV 或 JSON 文件"
    )
    parser.add_argument(
        "--input_type",
        choices=["csv", "json"],
        required=True,
        help="输入文件类型",
    )
    parser.add_argument(
        "--text_column",
        default="微博正文",
        help="文本字段名",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="显示前 K 个主题"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="保存 doc_topics.json（可选）",
    )

    args = parser.parse_args()

    # 读取文本
    if args.input_type == "csv":
        texts = load_csv(
            args.input, text_column=args.text_column
        )
    else:
        raw_json = load_json(args.input)
        texts = [
            doc.get(args.text_column, "").strip()
            for doc in raw_json
            if doc.get(args.text_column)
        ]

    print(f"[INFO] 共读取 {len(texts)} 条文本")

    all_doc_topics = infer_texts(
        texts=texts,
        lda_model_path=args.model,
        vocab_path=args.vocab,
        topics_json_path=args.topics,
        top_k=args.top_k,
    )

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                all_doc_topics,
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n[INFO] 推理结果已保存到 {args.output}")
