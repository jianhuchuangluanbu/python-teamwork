import nltk
from nltk.tokenize import word_tokenize
from test_cases import word_tokenize_cases

# nltk.download('punkt') 使用前需要先下载punkt模型
# Punkt模型是一个基于无监督学习的句子边界检测工具，专门用于句子分割

for case in word_tokenize_cases:
    text = case["text"]
    result = word_tokenize(text)
    print(result)

# result：
# ['The', 'international', 'community', 'must', 'continue', 'to', 'pay', 'close', 'attention', 'to', 'the', 'troubling', 'human', 'rights', 'situation', 'in', 'the', 'Democratic', 'People', '’', 's', 'Republic', 'of', 'Korea', '(', 'DPRK', ')', 'and', 'find', 'ways', 'to', 'revive', 'dialogue', 'with', 'the', 'Government', ',', 'the', 'UN', 'Security', 'Council', 'heard', 'on', 'Wednesday', '.']
# ['直面事实、兑现承诺——古特雷斯呼吁世界就气候变化立即采取行动']
# ['报告提出了一项计划，旨在改革全球金融体系，推动联合国可持续发展目标刺激计划，以应对当前的债务危机。建议采取的措施包括：改善发展中国家对全球金融体系治理的有效参与；通过有效的债务解决机制解决债务成本上升和债务困扰的风险；扩大应急融资，以便在危机时期提供更大的流动性，这样各国就不会在万不得已的情况下被迫举债；通过动员多边开发银行和私人资源，大规模增加负担得起的长期融资。', ']']