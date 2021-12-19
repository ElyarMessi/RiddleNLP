# NLP Riddle Guessing



### 12.17 伊力亚尔

首先处理了一下数据，把**一个谜语，五个候选项的一个样本**，变成**一个谜语，一个候选项的五个样本**



写了个data_loader，整合了wiki的数据，一个样本包括以下几个项：

**谜语、提示、候选项、候选项的wiki解释、标签**



其中，目前对bert的输入，是简单的句子对儿

第一项是谜语提示拼接谜语，第二项是候选项拼接候选项的wiki解释

后续可以调整这一块



写了最简单的模型设计，能跑

bert输入直接去pool过后的结果接上两个全连接层

这一块也是后需调整的重点叭



写了个train的初步，把取数据那块给写上了

后续补充loss，optimizer啥的


### Explore
#### 1 binary classification * 5
```text
[CLS] + [TIP_TOKEN1, TIP_TOKEN2, ...] + [FACE_TOKEN1, FACE_TOKEN2, ...] + [SEP] + [CAN_TOKEN1, CAN_TOKEN2, ...] + [WIKI_TOKEN1, WIKI_TOKEN2]
```
Unbalanced Label: Led to all-false prediction.

