import nltk
from nltk.corpus import twitter_samples # nltk自带的语料库，包含推文样本，用于情感分析
from nltk.tokenize import word_tokenize # 分词器，字符串分割成单独的词语
from nltk.corpus import stopwords   # 停用词，过滤常用词如语气词、连接词等
from nltk.stem import WordNetLemmatizer # 词形还原，将词语转变为基本形式
import string   # 去标点
import itertools    # 迭代器，文本处理中用于展开列表
from nltk import FreqDist   # 计算词语或其他元素在文本中出现的频率，统计词频分布
from sklearn.model_selection import train_test_split    # 将数据集划分为训练集和测试集，随机分割数据集，保持训练时数据的独立性和随机性
from nltk.classify import NaiveBayesClassifier  # 朴素贝叶斯分类器
from nltk.classify.util import accuracy as nltk_accuracy    # 计算分类器准确率

# 下载必要的资源
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 加载数据
positive_tweets = twitter_samples.strings('positive_tweets.json')   # strings方法读取文件，并返回一个包含所有正面推文的列表，每条推文作为列表中的一个字符串元素.结果赋值给一个字符串列表
negative_tweets = twitter_samples.strings('negative_tweets.json')   # 负面

# 预处理函数
stop_words = set(stopwords.words('english'))    #在 twitter上下载的推文样本是英文，因此停用词处理选择对英文进行处理，过滤像‘is’、’and‘、’the‘这类词语
lemmatizer = WordNetLemmatizer()    # 将词语进行还原为基本形式，’doing‘->'do'。


# 对推文进行预处理
def preprocess_tweet(tweet):
    tokens = word_tokenize(tweet)
    tokens = [token.lower() for token in tokens if token.isalpha()] #转换成小写且仅保留字母
    tokens = [token for token in tokens if token not in stop_words] #去除停用词
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  #词形还原
    return tokens


# 处理数据
positive_tokens = [preprocess_tweet(tweet) for tweet in positive_tweets]
negative_tokens = [preprocess_tweet(tweet) for tweet in negative_tweets]

# 特征提取
all_tokens = list(itertools.chain(*positive_tokens)) + list(itertools.chain(*negative_tokens))
freq_dist = FreqDist(all_tokens)    # 计算列表中每个词汇的出现频率，并赋值给freq_dist,获取词汇及其出现次数的信息
vocab = list(freq_dist.keys())  # 将词汇列表转换为一个普通的Python列表

# vocab即为经过处理得到的模型特征的基础词汇表

# 提取特征
def extract_features(tokens):
    features = {}   # 创建特征字典
    token_set = set(tokens)
    for word in vocab:
        features[word] = (word in token_set)
    return features


positive_features = [extract_features(tokens) for tokens in positive_tokens]
negative_features = [extract_features(tokens) for tokens in negative_tokens]

# 构建标签数据
positive_labels = [(feature, 'positive') for feature in positive_features]
negative_labels = [(feature, 'negative') for feature in negative_features]

# 合并数据并拆分训练集和测试集
dataset = positive_labels + negative_labels
train_data, test_data = train_test_split(dataset, test_size=0.25)   # 25%作为测试集，75%作为训练集

# 训练模型
model = NaiveBayesClassifier.train(train_data)

# 评估模型
print(f"Accuracy: {nltk_accuracy(model, test_data):.2f}")   # 输出测试集准确率
model.show_most_informative_features(10)    # 显示最具信息性的特征（前10个）


'''
'''
