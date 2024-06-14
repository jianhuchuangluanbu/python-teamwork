import nltk
from nltk.tokenize import sent_tokenize
from test_cases import cases

# sent_tokenize可以将一段文本拆分成句子。

for case in cases:
    text = case["text"]
    result = sent_tokenize(text)
    print(result)

# results:
# ['The international community must continue to pay close attention to the troubling human rights situation in the Democratic People’s Republic of Korea (DPRK) and find ways to revive dialogue with the Government, the UN Security Council heard on Wednesday.']
# ['Born to a leading family in the capital, Pyongyang, Mr. Kim was 19 when he left to study in Beijing in 2010.', 'Using the internet, he said he learned about his homeland and “the horrific truth” previously hidden to him.']
# ['He too welcomed the OECD figures announced on Wednesday and said there is now an opportunity to consider what the transition to renewable energy really means for SIDS.', 'It amounts to economic transformation, he said.']