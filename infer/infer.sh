python infer.py \
  --model /home/liuyuan/Class_data/model/lda.model \
  --vocab /home/liuyuan/Class_data/data/IT-IDF/vocab_tfidf.json\
  --topics /home/liuyuan/Class_data/model/topics.json \
  --input /home/liuyuan/Class_data/infer/test_data/test.csv \
  --input_type csv \
  --text_column 微博正文 \
  --top_k 5 \
  --output /home/liuyuan/Class_data/infer/test_result/doc_topics.json