# 项目使用说明

## 目录结构概览
```
.
├── content_info/ # 文档分词信息
├── core/ # 分词脚本用到的相关路径
├── data/ # 数据处理目录
│ ├── process_1/ # 中间处理 JSON 文件
│ └── IT-IDF/ # TF-IDF 处理结果
├── infer/ # 推理脚本和输出
├── model/ # OLDA/LDA 模型及结果
├── tokenier/ # 分词相关脚本
├── vocab/ # 词表及停用词相关
└── OLDA/ # OLDA 训练脚本

```

## 环境依赖

- Python 3.11+
- 主要依赖库：
  - pandas
  - jieba
  - tinydb
  - gensim
  - tqdm
- 可通过 `requirements.txt` 安装：

```bash
pip install -r requirements.txt
```
数据处理及训练流程
1️⃣ 文本分词处理
脚本：tokenier/jieba_cut.py

功能：

对原始 CSV 文档进行分词处理

去除停用词和标点符号

保存处理结果：

输出文件：

content_info/content.json → 每条文档分词及词频信息

vocab/vocab.json → 累积词表及词频
运行方式：



```bash 
python tokenier/jieba_cut.py
```
jieba_cut.py使用了vocab/stopwords.txt进行了初步的停用词过滤，一开始本来打算一边切词一边过滤，但是发现这样额度得到的结果很差，停用词词量太小，很多无关词无法过滤，所以又用了其他的停用词表合并了之后继续处理
注意：jieba_cut.py 会引用 core/config.py 中的路径配置。

2️⃣ 停用词过滤
脚本：vocab/filter_stop.sh → 内部调用 vocab/json_structure.py
使用了合并了多个停用词集的停用词词表vocab/stop_merge.json
功能：

根据停用词表过滤 vocab.json 中的无效词

生成中间处理 JSON 文件，存放在：
```
data/process_1/
  ├── vocab.json
  └── freq.json

```
运行方式：
bash
```bash
sh vocab/filter_stop.sh
```
3️⃣ TF-IDF 处理
脚本：vocab/tf_idf.sh → 调用 vocab/TF-IDF.py

功能：

基于 data/process_1 中的 vocab.json 和 freq.json 计算 TF-IDF

输出结果保存在：

```
data/IT-IDF/
  ├── vocab_tfidf.json
  └── freq_tfidf.json

```
运行方式：
```bash
sh vocab/tf_idf.sh
```
4️⃣ OLDA/LDA 训练
脚本：OLDA/OLDA.sh → 调用 OLDA/OLDA.py

功能：

使用 data/IT-IDF/vocab_tfidf.json 和 freq_tfidf.json 训练 OLDA/LDA 模型

输出模型文件和训练结果到 model/：
```
model/
  ├── lda.model                  # 模型文件，可直接 load
  ├── lda.model.state            # 内部训练状态
  ├── lda.model.expElogbeta.npy  # 主题-词概率矩阵
  ├── lda.model.id2word          # id2word 映射
  ├── topics.json                # 主题关键词
  └── doc_topics.json            # 文档主题分布
``` 
运行方式：

```bash

sh OLDA/OLDA.sh
```
5️⃣ 推理 / 文档主题分析
脚本：infer/infer.py

功能：

对新文档进行分词、BOW 映射

使用训练好的 model/lda.model 做主题推理

输出文档主题分布，可保存为 JSON


##注意
因为我在写代码的时候省事，每一个sh执行脚本用了绝对路径，需要跟据自己的路径进行修改，我将模型和所有结果都进行了上传，可以直接进行使用，其他的脚本其实不需要咋需要了


---

````markdown
# LDA 文档主题推理（infer.py）

本模块用于基于已训练好的 LDA 模型，对新的文本数据进行主题分布推理，并输出每篇文档的主题概率结果（`doc_topics.json`），供后续可视化或分析使用。

支持 **CSV / JSON** 两种输入格式，适用于中文文本（基于 `jieba` 分词）。

---

## 1. 功能概述

- 加载已训练的 `gensim` LDA 模型
- 使用训练阶段保存的词表（`vocab.json`）
- 对输入文本进行分词并构造 BOW
- 推理每篇文档的主题分布
- 可选打印主题关键词
- 将结果保存为 JSON（用于可视化）

---

## 2. 目录结构示例

```text
Class_data/
├── model/
│   ├── lda.model
│   ├── topics.json
├── data/
│   └── IT-IDF/
│       └── vocab_tfidf.json
├── infer/
│   ├── infer.py
│   ├── test_data/
│   │   └── test.csv
│   └── test_result/
│       └── doc_topics.json
````

---



```bash
pip install gensim jieba numpy
```

---

## 4. 输入数据说明

### 4.1 CSV 输入格式

CSV 文件需包含一列文本字段，例如：

| 微博正文        |
| ----------- |
| 今天讨论人工智能的发展 |
| 大模型在工业领域的应用 |

默认字段名为 `微博正文`，可通过参数指定。

---

### 4.2 JSON 输入格式

JSON 文件应为列表结构，例如：

```json
[
  {"微博正文": "今天讨论人工智能的发展"},
  {"微博正文": "大模型在工业领域的应用"}
]
```

---

## 5. 词表说明（vocab.json）

词表文件格式为：

```json
{
  "0": "中国",
  "1": "人工智能",
  "2": "模型"
}
```

> 注意：
>
> key 为 字符串形式的 token id
> value 为对应的词

该词表必须与 LDA 训练阶段保持一致。

---

## 6. 主题关键词文件（topics.json，可选）

用于在命令行中打印每个主题的关键词。

示例格式：

```json
{
  "0": ["人工智能", "模型", "算法"],
  "1": ["经济", "市场", "企业"]
}
```

---

## 7. 使用方法（命令行）

### 7.1 基本用法

```bash
python infer.py \
  --model /home/liuyuan/Class_data/model/lda.model \
  --vocab /home/liuyuan/Class_data/data/IT-IDF/vocab_tfidf.json \
  --topics /home/liuyuan/Class_data/model/topics.json \
  --input /home/liuyuan/Class_data/infer/test_data/test.csv \
  --input_type csv \
  --text_column 微博正文 \
  --top_k 5 \
  --output /home/liuyuan/Class_data/infer/test_result/doc_topics.json
```

---

## 8. 参数说明

| 参数              | 说明                  |
| --------------- | ------------------- |
| `--model`       | 已训练好的 `lda.model`   |
| `--vocab`       | 词表文件（id → word）     |
| `--topics`      | 主题关键词文件（可选）         |
| `--input`       | 输入 CSV / JSON 文件    |
| `--input_type`  | 输入类型：`csv` 或 `json` |
| `--text_column` | 文本字段名（默认：微博正文）      |
| `--top_k`       | 打印前 K 个主题           |
| `--output`      | 推理结果输出路径（JSON）      |

---

## 9. 输出结果说明（doc_topics.json）

输出为一个列表，每个元素对应一篇文档：

```json
[
  [[0, 0.4231], [3, 0.2178], [1, 0.1056]],
  [[2, 0.5123], [0, 0.1984]]
]
```

格式说明：

```text
[
  [topic_id, probability],
  ...
]
```

* `topic_id`：主题编号
* `probability`：该主题在文档中的概率

---

## 10. 注意事项

* 若文本中所有词均不在训练词表中，该文档将返回空主题列表
* 推理阶段不会重新训练模型
* `jieba` 临时缓存目录已设置为：

  ```
  /home/liuyuan/.jieba_cache
  ```



````markdown
# LDA 文档主题可视化（view.py）

本模块用于基于 `doc_topics.json` 和 `topics_labeled.json` 对 LDA 推理结果进行可视化，包括：

- 全局主题分布柱状图
- 单篇文档主题分布柱状图（前 K 个主题）
- 文档—主题空间 t-SNE 投影
- 主题词云（可选单独生成）

支持中文显示，已集成自定义字体（SimHei.ttf）。


## 2. 文件目录示例

```text
Class_data/
├── infer/
│   └── test_result/
│       └── doc_topics.json
├── model/
│   └── topics_labeled.json
├── view/
│   ├── view.py
│   └── fonts/
│       └── SimHei.ttf
└── view_png/
    └── 输出的图表文件夹
```

---

## 3. view.py 功能说明

### 3.1 全局主题分布

绘制所有文档的主题概率累积柱状图，保存为：

```
output_dir/global_topic_distribution.png
```

### 3.2 单篇文档主题分布

绘制每篇文档前 K 个主题柱状图，保存为：

```
output_dir/doc_{doc_idx}_top{top_k}_topics.png
```

### 3.3 文档—主题空间 t-SNE 投影

将文档在主题空间降维到二维，用散点图显示文档间的关系：

```
output_dir/doc_tsne.png
```

### 3.4 主题词云（可选）

根据 `topics_labeled.json` 中的关键词生成词云，保存到单独目录。

---

## 4. 命令行使用示例

### 4.1 普通可视化

```bash
DOC_TOPICS="/home/liuyuan/Class_data/infer/test_result/doc_topics.json"
TOPICS_LABELED="/home/liuyuan/Class_data/model/topics_labeled.json"
VIEW_OUTPUT="/home/liuyuan/Class_data/view/view_png"
DOC_OUTPUT="/home/liuyuan/Class_data/view/view_png/view_doc"

mkdir -p "${VIEW_OUTPUT}"

python view.py \
  --doc_topics "${DOC_TOPICS}" \
  --topics_labeled "${TOPICS_LABELED}" \
  --output_dir "${VIEW_OUTPUT}" \
  --output_doc "${DOC_OUTPUT}"
```

说明：

* `--doc_topics`：LDA 推理输出 JSON
* `--topics_labeled`：主题标签及关键词 JSON
* `--output_dir`：图表输出目录
* `--output_doc`：单篇文档主题柱状图输出文件前缀（可选）

### 4.2 生成主题词云（可选）

```bash
WORDCLOUD_OUTPUT="/home/liuyuan/Class_data/view/view_word_clouds"
mkdir -p "${WORDCLOUD_OUTPUT}"

python - <<EOF
import json
from view import plot_all_topic_wordclouds

with open("${TOPICS_LABELED}", "r", encoding="utf-8") as f:
    topics_labeled = json.load(f)

plot_all_topic_wordclouds(
    topics_labeled,
    output_dir="${WORDCLOUD_OUTPUT}"
)
EOF
```

---

## 5. 输出文件示例

```
view_png/
├── global_topic_distribution.png
├── doc_0_top3_topics.png
├── doc_1_top3_topics.png
├── ...
└── doc_tsne.png

view_word_clouds/
├── topic_0.png
├── topic_1.png
├── ...
```

---

## 6. 注意事项

* 确保 `fonts/SimHei.ttf` 存在，否则中文字体显示会失败
* 如果只想生成词云，可单独调用 `plot_all_topic_wordclouds` 函数
* 每次运行会覆盖相同输出目录下同名文件
* `--output_doc` 可指定单篇文档柱状图输出路径前缀，用于批量保存

---

## 7. 可选改进

* 调整 `top_k` 参数绘制更多主题
* 配合 `infer.py` 的 `doc_topics.json` 生成完整实验流程
* 将输出目录统一管理，便于项目结构清晰
相关例子

## 可视化结果

### 全局主题分布
![全局主题分布](view/view_png/global_topic_distribution.png)

### 文档主题 t-SNE
![t-SNE](view/view_png/doc_tsne_by_topic_labeled.png)

### 主题词云
![Topic 0](view/view_word_clouds/topic_0.png)




