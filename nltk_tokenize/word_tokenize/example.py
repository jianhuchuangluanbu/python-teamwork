import nltk
from nltk.tokenize import word_tokenize
from test_cases import cases
from test_cases_cn import cases_cn

# nltk.download('punkt') 使用前需要先下载punkt模型
# Punkt模型是一个基于无监督学习的句子边界检测工具，专门用于句子分割

for case in cases:
    text = case["text"]
    result = word_tokenize(text)
    print(result)

# result：
# ['The', 'international', 'community', 'must', 'continue', 'to', 'pay', 'close', 'attention', 'to', 'the', 'troubling', 'human', 'rights', 'situation', 'in', 'the', 'Democratic', 'People', '’', 's', 'Republic', 'of', 'Korea', '(', 'DPRK', ')', 'and', 'find', 'ways', 'to', 'revive', 'dialogue', 'with', 'the', 'Government', ',', 'the', 'UN', 'Security', 'Council', 'heard', 'on', 'Wednesday', '.']
# ['China', 'and', 'Russia', 'opposed', 'the', 'meeting', 'and', 'called', 'for', 'a', 'procedural', 'vote', 'by', 'the', '15', 'members', ',', 'which', 'was', 'defeated', '.']
# ['One', 'consequence', 'is', 'that', 'divided', 'families', 'are', 'even', 'more', 'divided', '.', 'No', 'departures', 'means', 'no', 'reunification', 'with', 'families', 'abroad', '.']

# nltk不支持中文词性标注，如需对中文进行处理，需要借助jieba分词，再用nltk进行其他处理

import jieba

for case in cases_cn:
    text = case["text"]
    result = jieba.lcut(text)
    print(result)

# result:
# ['公共', '债务', '包括', '国内', '和', '国外', '的', '一般', '政府', '借款', '。', '这份', '题为', '“', '2024', '年', '的', '债务', '世界', ':', '全球', '繁荣', '日益增长', '的', '负担', '”', '的', '报告', '指出', '，', '特别', '是', '在', '非洲', '，', '多重', '全球', '危机', '后', '经济', '疲软', '导致', '债务', '负担', '加重', '。', '2013', '年', '至', '2023', '年间', '，', '债务', 'GDP', '比率', '超过', '60%', '的', '非洲', '国家', '从', '6', '个', '增加', '到', '27', '个', '。']
# ['报告', '提出', '了', '一项', '计划', '，', '旨在', '改革', '全球', '金融体系', '，', '推动', '联合国', '可', '持续', '发展', '目标', '刺激', '计划', '，', '以', '应对', '当前', '的', '债务', '危机', '。', '建议', '采取', '的', '措施', '包括', '：', '改善', '发展中国家', '对', '全球', '金融体系', '治理', '的', '有效', '参与', '；', '通过', '有效', '的', '债务', '解决', '机制', '解决', '债务', '成本上升', '和', '债务', '困扰', '的', '风险', '；', '扩大', '应急', '融资', '，', '以便', '在', '危机', '时期', '提供', '更大', '的', '流动性', '，', '这样', '各国', '就', '不会', '在', '万不得已', '的', '情况', '下', '被迫', '举债', '；', '通过', '动员', '多边', '开发', '银行', '和', '私人', '资源', '，', '大规模', '增加', '负担得起', '的', '长期', '融资', '。']