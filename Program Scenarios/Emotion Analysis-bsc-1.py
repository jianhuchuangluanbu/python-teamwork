import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import itertools
from nltk import FreqDist
from sklearn.model_selection import train_test_split
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

# 下载必要的资源
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 加载数据
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# 预处理函数
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_tweet(tweet):
    tokens = word_tokenize(tweet)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


# 数据增强函数
def augment(tokens, n=5):
    augmented_data = []
    for _ in range(n):
        augmented_tokens = []
        for token in tokens:
            augmented_token = token + ''.join([random.choice(string.punctuation) for _ in range(random.randint(1, 3))])
            augmented_tokens.append(augmented_token)
        augmented_data.append(augmented_tokens)
    return augmented_data


# 处理数据
positive_tokens = [preprocess_tweet(tweet) for tweet in positive_tweets]
negative_tokens = [preprocess_tweet(tweet) for tweet in negative_tweets]

# 数据增强
augmented_positive_tokens = list(itertools.chain(*[augment(tokens) for tokens in positive_tokens]))
augmented_negative_tokens = list(itertools.chain(*[augment(tokens) for tokens in negative_tokens]))

# 特征提取
all_tokens = list(itertools.chain(*positive_tokens)) + list(itertools.chain(*negative_tokens)) + \
             list(itertools.chain(*augmented_positive_tokens)) + list(itertools.chain(*augmented_negative_tokens))
freq_dist = FreqDist(all_tokens)
vocab = list(freq_dist.keys())


def extract_features(tokens):
    features = {}
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
train_data, test_data = train_test_split(dataset, test_size=0.25)

# 训练模型
model = NaiveBayesClassifier.train(train_data)

# 评估模型
print(f"Accuracy: {nltk_accuracy(model, test_data):.2f}")
model.show_most_informative_features(10)