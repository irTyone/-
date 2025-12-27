import argparse
import json
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import numpy as np
from matplotlib import font_manager, rcParams
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
FONT_PATH = os.path.join(
    os.path.dirname(__file__),
    "fonts",
    "SimHei.ttf"
)

if not os.path.exists(FONT_PATH):
    raise RuntimeError(f"❌ 找不到中文字体文件: {FONT_PATH}")

# 把字体注册进 matplotlib
font_manager.fontManager.addfont(FONT_PATH)

# 获取 matplotlib 认可的字体 family 名
font_prop = font_manager.FontProperties(fname=FONT_PATH)
font_name = font_prop.get_name()


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = [font_name]
rcParams["axes.unicode_minus"] = False

print(f"[INFO] Matplotlib 正在使用字体: {font_name}")
def plot_global_topic_distribution(all_doc_topics, topics_labeled, output_dir="output"):
    """
    绘制全局主题分布柱状图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 累计所有文档主题概率
    topic_sums = {}
    for doc in all_doc_topics:
        for tid, prob in doc:
            topic_sums[tid] = topic_sums.get(tid, 0) + prob
    
    # 排序
    sorted_topics = sorted(topic_sums.items(), key=lambda x: x[1], reverse=True)
    labels = [f"{tid} {topics_labeled[str(tid)]['label']}" for tid, _ in sorted_topics]
    values = [v for _, v in sorted_topics]

    plt.figure(figsize=(10, 8))
    plt.barh(labels[::-1], values[::-1])
    plt.xlabel("累积概率")
    plt.title("全局主题分布")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_topic_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 全局主题分布已保存到 {output_dir}/global_topic_distribution.png")


def plot_single_doc(all_doc_topics, doc_idx, topics_labeled, top_k=3, output_dir="output"):
    """
    绘制单篇文档主题分布柱状图，只显示前 top_k 个主题
    """
    os.makedirs(output_dir, exist_ok=True)
    if doc_idx >= len(all_doc_topics):
        print(f"[WARN] 文档索引 {doc_idx} 超出范围")
        return

    doc = all_doc_topics[doc_idx]
    doc = sorted(doc, key=lambda x: x[1], reverse=True)[:top_k]
    labels = [f"{tid} {topics_labeled[str(tid)]['label']}" for tid, _ in doc]
    values = [v for _, v in doc]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylabel("概率")
    plt.title(f"文档 {doc_idx} 前 {top_k} 主题分布")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"doc_{doc_idx}_top{top_k}_topics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 文档 {doc_idx} 前 {top_k} 主题分布已保存到 {output_dir}/doc_{doc_idx}_top{top_k}_topics.png")


def plot_all_topic_wordclouds(topics_labeled, output_dir):
    """
    为所有主题绘制词云
    topics_labeled:
    {
      topic_id: {
        "label": str,
        "keywords": [word1, word2, ...]
      }
    }
    """
    os.makedirs(output_dir, exist_ok=True)

    font_path = os.path.join(
    os.path.dirname(__file__),
    "fonts",
    "SimHei.ttf"
)

    if not os.path.exists(font_path):
        raise RuntimeError(f"❌ 找不到中文字体文件: {font_path}")

    for topic_id, topic_info in topics_labeled.items():
        keywords = topic_info.get("keywords", [])

        if not keywords:
            print(f"[WARN] 主题 {topic_id} 没有关键词，跳过")
            continue

        text = " ".join(keywords)

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            font_path=font_path
        ).generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"主题 {topic_id}: {topic_info.get('label', '')}")

        save_path = os.path.join(output_dir, f"topic_{topic_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"[INFO] 主题 {topic_id} 词云已保存到 {save_path}")

def plot_tsne(
    all_doc_topics,
    topics_labeled,
    output_dir="output",
    label_points_n=10,   # 标注前 N 个文档编号
):
    """
    文档—主题空间 t-SNE 投影
    - 按主导主题着色
    - 每个主题显示中心标签
    - 可选：少量点编号
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------
    # 1. 构造 文档 × 主题 矩阵
    # ---------------------------
    n_docs = len(all_doc_topics)
    n_topics = max(tid for doc in all_doc_topics for tid, _ in doc) + 1

    X = np.zeros((n_docs, n_topics))
    dominant_topics = []

    for i, doc in enumerate(all_doc_topics):
        if not doc:
            dominant_topics.append(-1)
            continue

        for tid, prob in doc:
            X[i, tid] = prob

        # 主导主题 = 概率最大的主题
        dominant_topics.append(max(doc, key=lambda x: x[1])[0])


    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, n_docs - 1),
        init="pca",
        learning_rate="auto",
    )
    X_embedded = tsne.fit_transform(X)

   
    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=dominant_topics,
        cmap="tab20",
        s=18,
        alpha=0.8,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("主导主题 ID")


    topic_points = {}

    for i, tid in enumerate(dominant_topics):
        if tid < 0:
            continue
        topic_points.setdefault(tid, []).append(X_embedded[i])

    for tid, points in topic_points.items():
        points = np.array(points)
        center = points.mean(axis=0)

        topic_label = topics_labeled.get(str(tid), {}).get("label", f"Topic {tid}")

        plt.text(
            center[0],
            center[1],
            topic_label,
            fontsize=10,
            weight="bold",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="gray",
                alpha=0.8,
                boxstyle="round,pad=0.3",
            ),
        )

 
    for i in range(min(label_points_n, n_docs)):
        plt.text(
            X_embedded[i, 0],
            X_embedded[i, 1],
            str(i),
            fontsize=8,
            color="black",
        )


    plt.title("文档主题空间 t-SNE（主导主题 + 中心标签）")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "doc_tsne_by_topic_labeled.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"[INFO] t-SNE 可视化已保存到 {output_dir}/doc_tsne_by_topic_labeled.png"
    )


def main():
    parser = argparse.ArgumentParser(description="OLDA 文档主题可视化")
    parser.add_argument("--doc_topics", required=True, help="doc_topics.json 文件")
    parser.add_argument("--topics_labeled", required=True, help="topics_labeled.json 文件")
    parser.add_argument("--output_dir", default="output", help="可选：输出文件夹")
    parser.add_argument("--doc_idx", type=int, default=0, help="可选：绘制单篇文档的索引")
    parser.add_argument("--output_doc", type=str, default=0, help="可选：单篇文档的主题预测输出文件")
    args = parser.parse_args()

    with open(args.doc_topics, "r", encoding="utf-8") as f:
        all_doc_topics = json.load(f)
    with open(args.topics_labeled, "r", encoding="utf-8") as f:
        topics_labeled = json.load(f)

    # 可视化
    plot_global_topic_distribution(all_doc_topics, topics_labeled, output_dir=args.output_dir)
    for i in range(len(all_doc_topics)):
        plot_single_doc(
        all_doc_topics,
        doc_idx=i,
        topics_labeled=topics_labeled,
        top_k=3,
        output_dir=args.output_doc
    )
    # plot_all_topic_wordclouds(topics_labeled, output_dir=args.output_dir)
    plot_tsne(all_doc_topics, topics_labeled,output_dir=args.output_dir)


if __name__ == "__main__":
    main()
