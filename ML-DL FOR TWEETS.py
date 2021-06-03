#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import re
import nltk
from tqdm import tqdm
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#loading the data
df=pd.read_csv('train.csv')
print(df.shape) 
print(df.head())
print(df.info())
df['label'].value_counts()

y_value_counts=df['label'].value_counts()
print("Racist tweets  = ",y_value_counts[1], "with percentage ", (y_value_counts[1]*100)/(y_value_counts[0]+y_value_counts[1]),'%')
print("Not Racist tweets  = ",y_value_counts[0], "with percentage ", (y_value_counts[0]*100)/(y_value_counts[0]+y_value_counts[1]),'%')


#lets see the classes through bar graph
data=dict(racist=y_value_counts[1],not_racist=y_value_counts[0])
cls=data.keys()
value=data.values()

plt.bar(cls,value,color='maroon',width=0.2)

df['tweet']=df['tweet'].str.replace(' ','_')
df['tweet']=df['tweet'].str.replace('-','_')
df['tweet']=df['tweet'].str.lower()

def expand(sent):
    "This function will replace english short notations with full form"
    
    sent=re.sub(r"can't", "can not",sent)
    sent=re.sub(r"won't", "will not",sent)
    
    sent=re.sub(r"n\'t", " not",sent)
    sent=re.sub(r"\'re", " are",sent)
    sent=re.sub(r"\'m"," am",sent)
    sent=re.sub(r"\'s"," is",sent)
    sent=re.sub(r"\'ll"," will",sent)
    sent=re.sub(r"\'ve"," have",sent)
    sent=re.sub(r"\'d"," would",sent)
    sent=re.sub(r"\'t", " not",sent)
    
    return sent

# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

def preprocess_tweet(text):
    "function for preprocess the text data"
    
    preprocessed_tweet=[]
    
    for sentence in tqdm(text):
        sent=expand(sentence)
        sent=sent.replace("\\r"," ")
        sent=sent.replace("\\n"," ")
        sent=sent.replace('\\"'," ")
        sent=re.sub("[^A-Za-z0-9]+"," ",sent)
        
        # https://gist.github.com/sebleier/554280
        sent=" ".join(i for i in sent.split() if i.lower() not in stopwords)
        preprocessed_tweet.append(sent.lower().strip())
        
    return preprocessed_tweet
        
preprocessed_tweets=preprocess_tweet(df['tweet'].values)

df['tweet']=preprocessed_tweets

df["tweet"][10]

y=df['label']
x=df.drop(['label'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=40)

vect=TfidfVectorizer(min_df=10)

vect.fit(x_train['tweet'].values)

train_tweet=vect.transform(x_train['tweet'].values)
test_tweet=vect.transform(x_test['tweet'].values)

print(train_tweet.shape,y_train.shape)
print(test_tweet.shape,y_test.shape)

#calculating sentiment scores for train data
x_train_sent=np.ndarray.tolist(x_train["tweet"].values)

sia=SentimentIntensityAnalyzer()
ps=[]
for i in range(len(x_train_sent)):
    ps.append((sia.polarity_scores((x_train_sent[i]))))
    
x_train_polarity=np.array(ps)
x_train_polarity=x_train_polarity.reshape(-1,1)
x_train_polarity.shape

#storing only scores of sentiment
x_t=[]
for i in range(len(x_train)):
    for j in x_train_polarity[0][0]:
        x_t.append(x_train_polarity[i][0][j])
x_t=np.array(x_t)
x_t=x_t.reshape(-1,4)
x_t.shape

#calculating sentiment scores for test data
x_test_sent=np.ndarray.tolist(x_test["tweet"].values)

sia=SentimentIntensityAnalyzer()
ps=[]
for i in range(len(x_test_sent)):
    ps.append((sia.polarity_scores((x_test_sent[i]))))
    
x_test_polarity=np.array(ps)
x_test_polarity=x_test_polarity.reshape(-1,1)
x_test_polarity.shape

#storing only scores of sentiment
x_tests=[]
for i in range(len(x_test)):
    for j in x_test_polarity[0][0]:
        x_tests.append(x_test_polarity[i][0][j])
x_tests=np.array(x_tests)
x_tests=x_tests.reshape(-1,4)
x_tests.shape

from scipy.sparse import hstack

x_tr=hstack((train_tweet,x_t))
x_te=hstack((test_tweet,x_tests))

print(x_tr.shape)
print(x_te.shape)

wt={0:1,1:5}  #since the data is imbalanced , we assign some more weight to class 1

# DECISION TREE METHOD

clf=DecisionTreeClassifier(class_weight=wt)

parameters=dict(max_depth=[1,5,10,50],min_samples_split=[5,10,100,500])

search=RandomizedSearchCV(clf,parameters,random_state=10)
result=search.fit(x_tr,y_train)
result.cv_results_

search.best_params_

cls = DecisionTreeClassifier(max_depth=50,min_samples_split=5,random_state=10,class_weight=wt)
cls.fit(x_tr,y_train)

y_pred_train=cls.predict(x_tr)
y_pred_test=cls.predict(x_te)

train_fpr,train_tpr,tr_treshold=roc_curve(y_train,y_pred_train)
test_fpr,test_tpr,te_treshold=roc_curve(y_test,y_pred_test)

train_auc=auc(train_fpr,train_tpr)
test_auc=auc(test_fpr,test_tpr)

plt.plot(train_fpr,train_tpr,label='Train AUC = '+str(train_auc))
plt.plot(test_fpr,test_tpr,label='Test AUC = '+str(test_auc))
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("AUC_Curve by Decision Tree")
plt.grid()
plt.show()

# SVM MODEL
from sklearn.svm import SVC
from sklearn import svm
svc = svm.SVC(kernel='rbf', C=1,gamma=0.1,probability=True).fit(x_tr,y_train)
print ("\n\n ---SVM Model---")

y_pred_train=svc.predict(x_tr)
y_pred_test=svc.predict(x_te)

train_fpr,train_tpr,tr_treshold=roc_curve(y_train,y_pred_train)
test_fpr,test_tpr,te_treshold=roc_curve(y_test,y_pred_test)

train_auc=auc(train_fpr,train_tpr)
test_auc=auc(test_fpr,test_tpr)

plt.plot(train_fpr,train_tpr,label='Train AUC = '+str(train_auc))
plt.plot(test_fpr,test_tpr,label='Test AUC = '+str(test_auc))
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("AUC_Curve by SVM")
plt.grid()
plt.show()

# Ada Boost
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
ada.fit(x_tr,y_train)

y_pred_train=ada.predict(x_tr)
y_pred_test=ada.predict(x_te)

train_fpr,train_tpr,tr_treshold=roc_curve(y_train,y_pred_train)
test_fpr,test_tpr,te_treshold=roc_curve(y_test,y_pred_test)

train_auc=auc(train_fpr,train_tpr)
test_auc=auc(test_fpr,test_tpr)

plt.plot(train_fpr,train_tpr,label='Train AUC = '+str(train_auc))
plt.plot(test_fpr,test_tpr,label='Test AUC = '+str(test_auc))
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("AUC_Curve by ADAP BOOSTING")
plt.grid()
plt.show()


# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=1000, 
    max_depth=None, 
    min_samples_split=10, 
    class_weight="balanced"
    #min_weight_fraction_leaf=0.02 
    )
rf.fit(x_tr,y_train)

y_pred_train=rf.predict(x_tr)
y_pred_test=rf.predict(x_te)

train_fpr,train_tpr,tr_treshold=roc_curve(y_train,y_pred_train)
test_fpr,test_tpr,te_treshold=roc_curve(y_test,y_pred_test)

train_auc=auc(train_fpr,train_tpr)
test_auc=auc(test_fpr,test_tpr)

plt.plot(train_fpr,train_tpr,label='Train AUC = '+str(train_auc))
plt.plot(test_fpr,test_tpr,label='Test AUC = '+str(test_auc))
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("AUC_Curve by Random Forest")
plt.grid()
plt.show()


def find_best_threshold(threshold, fpr, tpr):
    """it will give best threshold value that will give the least fpr"""
    t = threshold[np.argmax(tpr*(1-fpr))]
    
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    
    return t

def predict_with_best_t(proba, threshold):
    """this will give predictions based on best threshold value"""
    predictions = []
    for i in proba:
        if i>=threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

#computing confusion matrix for set_1

from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_treshold, train_fpr, train_tpr)
print("Train confusion matrix")
m_tr=(confusion_matrix(y_train, predict_with_best_t(y_pred_train, best_t)))
print(m_tr)
print("Test confusion matrix")
m_te=(confusion_matrix(y_test, predict_with_best_t(y_pred_test, best_t)))
print(m_te)

print(classification_report(y_test, y_pred_test))


vec=CountVectorizer(min_df=10)
vec.fit(x_train['tweet'].values)

#NAIVE BAYES

x_tr_count=vec.transform(x_train['tweet'].values)
x_te_count=vec.transform(x_test['tweet'].values)
x_tr_count.shape

x_tr_data=hstack((x_tr_count,x_t))
x_te_data=hstack((x_te_count,x_tests))

x_trn=scipy.sparse.csr_matrix(x_tr_count)
x_tst=scipy.sparse.csr_matrix(x_te_count)

from sklearn.naive_bayes import MultinomialNB

mod = MultinomialNB()
mod.fit(x_trn,y_train)

train_pred=mod.predict(x_trn)
test_pred=mod.predict(x_tst)

train_fpr,train_tpr,tr_treshold=roc_curve(y_train,train_pred)
test_fpr,test_tpr,te_treshold=roc_curve(y_test,test_pred)

train_auc=auc(train_fpr,train_tpr)
test_auc=auc(test_fpr,test_tpr)

plt.plot(train_fpr,train_tpr,label='Train AUC = '+str(train_auc))
plt.plot(test_fpr,test_tpr,label='Test AUC = '+str(test_auc))
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("AUC_Curve by Naive Bayes")
plt.grid()
plt.show()


#get the summary of this model

print(classification_report(y_test, test_pred))


#XGBOOST

from xgboost import XGBClassifier

xg=XGBClassifier()
param=dict(max_depth=[4,6,8,10],n_estimators=[100,500,1000,1500])
search=RandomizedSearchCV(xg,param,random_state=10)
srch=search.fit(x_tr,y_train)
srch.cv_results_

srch.best_estimator_

xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=None, monotone_constraints='()',
              n_estimators=500, n_jobs=8, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None).fit(x_tr, y_train)

prediction = xgb.predict(x_te) 

f1_score(y_test, prediction)


train_prediction=xgb.predict(x_tr)


train_fpr,train_tpr,tr_treshold=roc_curve(y_train,train_prediction)
test_fpr,test_tpr,te_treshold=roc_curve(y_test,prediction)

train_auc=auc(train_fpr,train_tpr)
test_auc=auc(test_fpr,test_tpr)

plt.plot(train_fpr,train_tpr,label='Train AUC = '+str(train_auc))
plt.plot(test_fpr,test_tpr,label='Test AUC = '+str(test_auc))
plt.legend()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("AUC_Curve")
plt.grid()
plt.show()

print(classification_report(y_test, prediction))

