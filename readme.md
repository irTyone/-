# 项目使用说明

## 目录结构概览

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

data/process_1/
  ├── vocab.json
  └── freq.json
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


data/IT-IDF/
  ├── vocab_tfidf.json
  └── freq_tfidf.json
运行方式：

```bash
sh vocab/tf_idf.sh
```
4️⃣ OLDA/LDA 训练
脚本：OLDA/OLDA.sh → 调用 OLDA/OLDA.py

功能：

使用 data/IT-IDF/vocab_tfidf.json 和 freq_tfidf.json 训练 OLDA/LDA 模型

输出模型文件和训练结果到 model/：
model/
  ├── lda.model                  # 模型文件，可直接 load
  ├── lda.model.state            # 内部训练状态
  ├── lda.model.expElogbeta.npy  # 主题-词概率矩阵
  ├── lda.model.id2word          # id2word 映射
  ├── topics.json                # 主题关键词
  └── doc_topics.json            # 文档主题分布
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