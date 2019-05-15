# ML
实现一些机器学习方法，numpy,或者调库函数


## barcode为条形码解析小实例
barcode.py

## resenet，tensorflow，keras，torch版本网络实现


## 最大熵 max_entroy.py

## ELMO中文版
ELMO 中文版

1.对文本数据进行分词

2.对分词进行统计，建立词表

3.对文件进行编码：word编码，char编码两种

4.对语言model的_build_word_char_embeddings()进行修改：
  由于之前是针对英文版本，所以对于char限制为261，对于中文来说一般大于261，所以此处要隐去。
