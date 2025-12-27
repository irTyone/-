#!/bin/bash


# =========================
# 路径配置
# =========================
DOC_TOPICS="/home/liuyuan/Class_data/infer/test_result/doc_topics.json"
TOPICS_LABELED="/home/liuyuan/Class_data/model/topics_labeled.json"
DOC_OUTPUT="/home/liuyuan/Class_data/view/view_png/view_doc"
# 普通可视化输出
VIEW_OUTPUT="/home/liuyuan/Class_data/view/view_png"

# 主题词云单独目录
WORDCLOUD_OUTPUT="/home/liuyuan/Class_data/view/view_word_clouds"


mkdir -p "${VIEW_OUTPUT}"
mkdir -p "${WORDCLOUD_OUTPUT}"


echo "[INFO] 开始生成主题可视化..."

python view.py \
  --doc_topics "${DOC_TOPICS}" \
  --topics_labeled "${TOPICS_LABELED}" \
  --output_dir "${VIEW_OUTPUT}"\
  --output_doc  "${DOC_OUTPUT}"

# =========================
# 单独再跑一次：只生成主题词云
# （如果你在 view.py 中是统一调用的，可删）
# =========================
# python - <<EOF
# import json
# from view import plot_all_topic_wordclouds

# with open("${TOPICS_LABELED}", "r", encoding="utf-8") as f:
#     topics_labeled = json.load(f)

# plot_all_topic_wordclouds(
#     topics_labeled,
#     output_dir="${WORDCLOUD_OUTPUT}"
# )
# EOF

