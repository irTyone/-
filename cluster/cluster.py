import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import Counter
from wordcloud import WordCloud

from matplotlib import font_manager, rcParams


# =========================
# 0. 中文字体配置（关键）
# =========================
FONT_PATH = "../view/fonts/SimHei.ttf"

if not os.path.exists(FONT_PATH):
    raise RuntimeError(f"❌ 找不到字体文件: {FONT_PATH}")

font_manager.fontManager.addfont(FONT_PATH)
font_prop = font_manager.FontProperties(fname=FONT_PATH)
font_name = font_prop.get_name()

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [font_name]
rcParams["axes.unicode_minus"] = False

print(f"[INFO] Matplotlib 使用字体: {font_name}")


# =========================
# 1. 文档-主题矩阵
# =========================
def build_doc_topic_matrix(doc_topics):
    n_docs = len(doc_topics)
    n_topics = max(t for doc in doc_topics for t, _ in doc) + 1

    X = np.zeros((n_docs, n_topics))
    for i, doc in enumerate(doc_topics):
        for tid, prob in doc:
            X[i, tid] = prob
    return X


# =========================
# 2. 聚类解释
# =========================
def explain_clusters(X, labels, topics_labeled, top_k_topics=3):
    explanations = {}

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        center = X[idx].mean(axis=0)
        top_topics = np.argsort(center)[::-1][:top_k_topics]

        explanations[int(c)] = {
            "topics": [(int(t), float(center[t])) for t in top_topics],
            "labels": [
                topics_labeled.get(str(t), {}).get("label", f"Topic {t}")
                for t in top_topics
            ],
            "keywords": sum(
                [topics_labeled.get(str(t), {}).get("keywords", []) for t in top_topics],
                []
            )
        }
    return explanations


# =========================
# 3. t-SNE + 文字说明
# =========================
def plot_tsne_with_labels(X, labels, explanations, out_path, title):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_emb = tsne.fit_transform(X)

    plt.figure(figsize=(9, 7))

    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(
            X_emb[idx, 0],
            X_emb[idx, 1],
            s=10,
            alpha=0.7,
            label=f"Cluster {c}"
        )

        # 计算 cluster 中心点
        cx, cy = X_emb[idx].mean(axis=0)

        # 构造说明文字
        topic_desc = " / ".join(
            f"{tid}:{lbl}"
            for (tid, _), lbl in zip(
                explanations[c]["topics"],
                explanations[c]["labels"]
            )
        )

        plt.text(
            cx,
            cy,
            f"Cluster {c}\n{topic_desc}",
            fontsize=9,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
        )

    plt.legend(fontsize=9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================
# 4. 迭代式 KMeans
# =========================
def iterative_kmeans(X, topics_labeled, k=8, max_iter=5, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    tsne_dir = os.path.join(out_dir, "iter_tsne")
    os.makedirs(tsne_dir, exist_ok=True)

    labels = None

    for i in range(max_iter):
        print(f"\n========== Iteration {i} ==========")

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        explanations = explain_clusters(X, labels, topics_labeled)

        plot_tsne_with_labels(
            X,
            labels,
            explanations,
            out_path=os.path.join(tsne_dir, f"iter_{i}.png"),
            title=f"第 {i} 轮聚类结果（K={k}）"
        )

    return labels, explanations


# =========================
# 5. 最终热点话题
# =========================
def generate_hot_topics(explanations, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    word_counter = Counter()

    for info in explanations.values():
        word_counter.update(info["keywords"])

    with open(os.path.join(out_dir, "final_hot_topics.json"), "w", encoding="utf-8") as f:
        json.dump(
            word_counter.most_common(50),
            f,
            ensure_ascii=False,
            indent=2
        )

    # 柱状图
    words, counts = zip(*word_counter.most_common(20))
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title("最终热点话题词 Top20")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hot_topics_bar.png"), dpi=150)
    plt.close()

    # 词云
    wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        font_path=FONT_PATH
    ).generate_from_frequencies(word_counter)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hot_topics_wordcloud.png"), dpi=150)
    plt.close()


# =========================
# 6. main
# =========================
if __name__ == "__main__":
    DOC_TOPICS = "../infer/test_result/doc_topics.json"
    TOPICS_LABELED = "../model/topics_labeled.json"
    OUTPUT_DIR = "./output"

    with open(DOC_TOPICS, "r", encoding="utf-8") as f:
        doc_topics = json.load(f)

    with open(TOPICS_LABELED, "r", encoding="utf-8") as f:
        topics_labeled = json.load(f)

    X = build_doc_topic_matrix(doc_topics)

    labels, explanations = iterative_kmeans(
        X,
        topics_labeled,
        k=8,
        max_iter=10,
        out_dir=OUTPUT_DIR
    )

    generate_hot_topics(explanations, OUTPUT_DIR)
