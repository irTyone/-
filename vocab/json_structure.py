import json
import argparse
from typing import Dict, List, Set



def load_stopwords(stopword_file: str) -> Set[str]:
   
    # ---------- JSON 停用词 ----------
    if stopword_file.endswith(".json"):
        with open(stopword_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "stops" in data:
            stops = data["stops"]
            if not isinstance(stops, list):
                raise ValueError("stop_merge.json 中 stops 必须是 list")
            return set(w.strip() for w in stops if isinstance(w, str) and w.strip())

        if isinstance(data, dict):
            return set(k.strip() for k in data.keys() if k.strip())

        if isinstance(data, list):
            return set(w.strip() for w in data if isinstance(w, str) and w.strip())

        raise ValueError("不支持的 stopwords JSON 格式")

    else:
        with open(stopword_file, "r", encoding="utf-8") as f:
            return set(w.strip() for w in f if w.strip())

def vocab_2_json(
    tinydb_file: str,
    output_file: str,
    stopwords: Set[str] | None = None
) -> Dict[str, str]:
    """
    TinyDB 格式词表 → 普通 JSON {word_id: word}
    同时过滤停用词
    """
    with open(tinydb_file, "r", encoding="utf-8") as f:
        vocab_tiny = json.load(f)

    vocab_new = {}

    for word_id, info in vocab_tiny["_default"].items():
        word = info["word"]

        # 停用词过滤
        if stopwords and word in stopwords:
            continue

        vocab_new[word_id] = word

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocab_new, f, ensure_ascii=False, indent=2)

    print(f"[OK] 词表保存到 {output_file}（保留 {len(vocab_new)} 个词）")
    return vocab_new



def freq_2_json(
    tinydb_file: str,
    output_file: str,
    vocab: Dict[str, str]
) -> List[Dict[str, int]]:
  
    # 构造反向映射：word -> word_id
    word2id = {word: wid for wid, word in vocab.items()}

    with open(tinydb_file, "r", encoding="utf-8") as f:
        docs_tiny = json.load(f)

    docs_new = []

    for doc_info in docs_tiny["_default"].values():
        words = doc_info.get("words", {})

        filtered = {}
        for word, freq in words.items():
            if word in word2id and freq > 0:
                filtered[word2id[word]] = freq

        if filtered:
            docs_new.append(filtered)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs_new, f, ensure_ascii=False, indent=2)

    print(f"[OK] 文档词频保存到 {output_file}（共 {len(docs_new)} 篇）")
    return docs_new



def main():
    parser = argparse.ArgumentParser(
        description="TinyDB → LDA/OLDA 可用 JSON（含停用词过滤）"
    )
    parser.add_argument("--vocab_tinydb", required=True, help="TinyDB 词表 JSON")
    parser.add_argument("--doc_tinydb", required=True, help="TinyDB 文档词频 JSON")
    parser.add_argument("--stopwords", required=True, help="停用词 txt 文件")
    parser.add_argument("--out_dir", required=True, help="输出目录")

    args = parser.parse_args()

    vocab_out = f"{args.out_dir}/vocab.json"
    freq_out = f"{args.out_dir}/freq.json"

  
    stopwords = load_stopwords(args.stopwords)
    print(f"[INFO] 加载停用词 {len(stopwords)} 个")

 
    vocab = vocab_2_json(
        args.vocab_tinydb,
        vocab_out,
        stopwords=stopwords
    )

    
    valid_vocab_ids = set(vocab.keys())
    freq_2_json(
    args.doc_tinydb,
    freq_out,
    vocab
)

    print("\n 更大规模的停用词筛查")


if __name__ == "__main__":
    main()