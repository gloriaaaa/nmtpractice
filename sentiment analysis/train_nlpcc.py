import pandas as pd
import jieba
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

#数据读取
df_weibo = pd.read_csv('data/nlpcc.csv',names=['data','label'],encoding='utf-8')

#读取全部词汇写入list
df_weibo = df_weibo.dropna()
content = df_weibo.data.values.tolist()
all_words_list = []
for line in content:
    a_words_list = jieba.lcut(line)
    if len(a_words_list) >= 1 and a_words_list != '\r\n':
        all_words_list.append(a_words_list)

df_text_words = pd.DataFrame({'data_cut':all_words_list})

#去除停用词
stopwords=pd.read_csv("stopwords.txt",index_col=False,sep="\n",quoting=3,names=['stopword'], encoding='utf-8')

def drop_stopwords(data_clean_before,stopwords):
    contents_clean = []
    all_words_clean = []
    for line in data_clean_before:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words_clean.append(str(word))
            
        if line_clean is None:
            line_clean.append("空")
        contents_clean.append(line_clean)
        #print(len(line_clean))
    return contents_clean,all_words_clean

	
data_clean_before = df_text_words.data_cut.values.tolist()    
stopwords = stopwords.stopword.values.tolist()
data_clean,all_words_clean = drop_stopwords(data_clean_before,stopwords)

#构造训练集
#df_text_words=pd.DataFrame({'data_clean':data_clean})
df_train=pd.DataFrame({'data_clean':data_clean,'label':df_weibo['label']})
#>>> label=df_weibo.label.values.tolist()
#>>> set(label)
#{'sadness', 'happiness', 'fear', 'disgust', 'none', 'surprise', 'anger', 'like'}
#标签映射
label_mapping = {"none": 1, "fear": 2, "sadness": 3, "happiness":4, "surprise": 5,"anger": 6,"like":7,"disgust":8}
df_train['label'] = df_train['label'].map(label_mapping)

#测试集数据集划分
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_train['data_clean'].values, df_train['label'].values, random_state=1)
#print(x_train.shape)
#print(y_train.shape)

#文本特征构造
words = []
for line_index in range(len(x_train)):
    try:
        words.append(' '.join(x_train[line_index]))
    except:
        #print(x_train[line_index])
        print("error!:"+str(line_index))
        x_train.drop(x_train.index[line_index],inplace=True)
        y_train.drop(y_train.index[line_index],inplace=True)
        #print (line_index)


#计算词频
#vec = CountVectorizer(analyzer='word', max_features=9000, lowercase = False)
#countresult = vec.fit(words)
#x = vec.transform(words)
#print(x.shape)

#训练分类器
#classifier = MultinomialNB()
#result = classifier.fit(x, y_train)
#构造测试集文本特征
test_words = []
for line_index in range(len(x_test)):
    try:
        test_words.append(' '.join(x_test[line_index]))
    except:
        print("error!:"+str(line_index))
        x_test.drop(x_test.index[line_index],inplace=True)
        y_test.drop(y_test.index[line_index],inplace=True)

#利用分类器测试
#print('test_words[0]',test_words[0])
#print('test_words_sorce',classifier.score(vec.transform(test_words), y_test))

#tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word', max_features=700, lowercase = False)
vectorizer.fit(words)


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)
print(classifier.score(vectorizer.transform(test_words), y_test))


#joblib.dump(countresult,'countresult.model')
#joblib.dump(result,'result.model')
    