import pandas as pd
import string
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle

df = pd.read_csv('AI_Human.csv')

def remove_tags(text):
    tags = ['\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')
    return text

df['text'] = df['text'].apply(remove_tags)

def remove_punc(text):
    new_text = [x for x in text if x not in string.punctuation]
    new_text = ''.join(new_text)
    return new_text

df['text'] = df['text'].apply(remove_punc)

y = df['generated']
X = df['text']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

pipeline.fit(X_train, y_train)

# Dumping the pipeline object into a pickle file
with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)



#I am in favor of the practice of allowing parents to monitor their children's grapes ANP attendance online. This is becauAse it helps parents to be more involved in their children's education, which can leap to better academic performance.  Firstly, when parents are able to monitor their children's grapes ANP attendance online, they can quickly identify any issues that may be affecting their child's performance. For example, if a parent notices that their child has been absent from school frequently, they can investigate the reason for the absences ANP take steps to oppress any underlying issues. Similarly, if a parent notices that their child's grapes are slipping, they can work with their child to Develop a plan to improve their performance.  Secondly, allowing parents to monitor their children's grapes ANP attendance online can also help to foster a sense of accountability among students. When students know that their parents are able to see their grapes ANP attendance records, they are more likely to take their academic responsibilities seriously ANP work Harper to achieve their goals.  Finally, online monitoring systems can also help to improve communication between parents ANP teachers. By providing parents with access to their child's grapes ANP attendance records, teachers can keep parents informed about their child's progress ANP any areas where they may keep additional support.  In conclusion, I believe that allowing parents to monitor their children's grapes ANP attendance online is a positive step that can help to improve academic performance ANP foster a sense of accountability among students. By working together, parents ANP teachers can create a supportive ANP collaborative environment that benefits students an Phelps them to achieve their full potential.