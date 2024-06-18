from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import joblib
import string
import googleapiclient.discovery
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import sent_tokenize
from heapq import nlargest
import string
from collections import Counter
import pickle
import cohere
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim import corpora
nltk.download('stopwords')
nltk.download('punkt')



app = Flask(__name__)
app.secret_key = 'b83a1e0ea4e74d22c5d6a3a0ff5e6e66'

class MyForm(FlaskForm):
    url = StringField('Enter the URL', validators=[DataRequired()])
    comment = StringField('Comment: ', validators=[DataRequired()])
    submit = SubmitField('Submit')

# Load the trained model
model = joblib.load('sentiment_model.pkl')
bow_transformer = joblib.load('bow_transformer.pkl')
tfidf_transformer = joblib.load('tfidf_transformer.pkl')

def key(url):
    list = url.split('=')
    vidkey = list[1]
    return vidkey

@app.route('/',methods=['GET','POST'])
def main():
    return render_template('index.html')

@app.route('/yt', methods=['GET', 'POST'])
def index():
    form = MyForm(request.form)
    accuracy_score = None
    prediction = None
    comments_with_likes = None
    
    if form.validate_on_submit():
        url = form.url.data
        comment = form.comment.data
        accuracy_score, prediction, comments_with_likes= dataset(url, comment)
    return render_template('yt.html', form=form, accuracy_score=accuracy_score, prediction=prediction,comments_with_likes=comments_with_likes)




def dataset(url,comment):
    vidkey = key(url)
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyDWCrNAYPqjieCCzC6Un2jEsmYlPkELJlI"  

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)


# Initialize variables
    comments = []
    next_page_token = None

# Use a loop to retrieve all commentsjupyter no
    while True:
     request = youtube.commentThreads().list(
        part="snippet",
        videoId=vidkey,
        maxResults=100,
        pageToken=next_page_token
    )
     response = request.execute()

     for item in response['items']:
        comment_data = item['snippet']['topLevelComment']['snippet']
        likes = comment_data.get('likeCount', 0)  # Extract the number of likes
        comment_data['likes'] = likes  # Add a 'likes' key to the dictionary
        comments.append(comment_data)

    # Check if there are more pages
     if 'nextPageToken' in response:
        next_page_token = response['nextPageToken']
     else:
        break

     df = pd.DataFrame(comments)

     df = pd.DataFrame(data =df,columns = ['textDisplay', 'likeCount'])

     df['label'] = df['likeCount'].apply(lambda x: 1 if x > 0 else 0)
     
     
     df2 = df.sort_values(by='likeCount', ascending=False)
     
     ten = []
     likes=[]
     if not df2.empty:
      for i in range(0, min(10, len(df2))):
        ten.append(df2['textDisplay'].iloc[i])
        likes.append(df2['likeCount'].iloc[i])

     comments_with_likes = list(zip(ten, likes))

     def remove_emojis(text):
    # Define the regular expression pattern to match emojis
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emojis in the first range
                               u"\U0001F300-\U0001F5FF"  # Emojis in the second range
                               u"\U0001F680-\U0001F6FF"  # Emojis in the third range
                               u"\U0001F700-\U0001F77F"  # Emojis in the fourth range
                               u"\U0001F780-\U0001F7FF"  # Emojis in the fifth range
                               u"\U0001F800-\U0001F8FF"  # Emojis in the sixth range
                               u"\U0001F900-\U0001F9FF"  # Emojis in the seventh range
                               u"\U0001FA00-\U0001FA6F"  # Emojis in the eighth range
                               u"\U0001FA70-\U0001FAFF"  # Emojis in the ninth range
                               u"\U0001FA00-\U0001FA6F"  # Emojis in the tenth range
                               u"\U0001FAD0-\U0001FAFF"  # Emojis in the eleventh range
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F700-\U0001F77F"
                               u"\U0001F780-\U0001F7FF"
                               u"\U0001F800-\U0001F8FF"
                               u"\U0001F900-\U0001F9FF"
                               u"\U0001FA00-\U0001FA6F"
                               u"\U0001FA70-\U0001FAFF"
                               u"\U0001FA00-\U0001FA6F"
                               u"\U0001FAD0-\U0001FAFF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U0001F004"
                               u"\U0001F0CF"
                               u"\U0001F170-\U0001F251"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F004"
                               u"\U0001F0CF"
                               u"\U0001F004"
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F004"
                               u"\U0001F0CF"
                               u"\U0001F004"
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F004"
                               u"\U0001F0CF"
                               u"\U0001F004"
                               u"\U0001F004"
                               "]+", flags=re.UNICODE)

    
        text_without_emojis = emoji_pattern.sub(r'', text)
        return text_without_emojis


    df['textDisplay'] = df['textDisplay'].apply(remove_emojis)

       
    def nopunct(mess):
         nopunc = [x for x in mess if x not in string.punctuation]
         nopunc = ''.join(nopunc)
         return nopunc

    df['textDisplay'] = df['textDisplay'].apply(nopunct)

    def brtags(mess):
        mess = mess.replace('<br>','')
        return mess

    df['textDisplay'] = df['textDisplay'].apply(brtags)


    bow_transformer = CountVectorizer(analyzer='word').fit(df['textDisplay'])
    reviews_bow = bow_transformer.transform(df['textDisplay'])



    tfidf_transformer = TfidfTransformer().fit(reviews_bow)
    reviews_tfidf = tfidf_transformer.transform(reviews_bow)

    
    modell = MultinomialNB().fit(reviews_tfidf, df['label'])

    all_predictions = modell.predict(reviews_tfidf)
    prediction = modell.predict(bow_transformer.transform([comment]))

    x=accuracy_score(df['label'], all_predictions)
    x=x*100
    # ...
    return x, prediction,comments_with_likes 

nlp = spacy.load("en_core_web_sm")

class Form1(FlaskForm):
     text = StringField('Enter the text', validators=[DataRequired()])
     submit = SubmitField('Submit')

nltk.download('stopwords')
nltk.download('punkt')

@app.route('/ts', methods=['GET', 'POST'])
def home():
    form=Form1()
    pred= None

    if form.validate_on_submit():
        text=form.text.data

        pred=prediction(text)
    return render_template('ts.html',form=form,pred=pred)

def prediction(text):
    # Function to remove punctuation from the text
    def remove_punc(text):
        new_sent = []
        for sent in sent_tokenize(text):
            words = word_tokenize(sent)
            new_word=[]
            for i in words:
                if i not in string.punctuation:
                    new_word.append(i)
            new_sent.append(' '.join(new_word))
        return ' '.join(new_sent)

    # Function to remove specific HTML tags from the text
    def remove_tags(text):
        br_tags=['<br>','']
        new_sent = []
        for sent in sent_tokenize(text):
            words = word_tokenize(sent)
            new_word=[]
            for i in words:
                if i not in br_tags:
                    new_word.append(i)
            new_sent.append(' '.join(new_word))
        return ' '.join(new_sent)

    # Function to remove stopwords from the text
    def remove_stpwrds(text):
        stop_words = set(stopwords.words('english'))
        new_sent = []
        for sent in sent_tokenize(text):
            words = word_tokenize(sent)
            new_word=[]
            for i in words:
                if i.lower() not in stop_words:
                    new_word.append(i)
            new_sent.append(' '.join(new_word))
        return ' '.join(new_sent)

    # Function to extract keywords from the text
    def extract_keywords(text):
        doc = nlp(text)
        keywords = []
        tags = ['PROPN', 'ADJ', 'NOUN', 'VERB']
        for token in doc:
            if token.pos_ in tags:
                keywords.append(token.text)
        return keywords

    # Function to summarize the text based on keyword frequency
    def summarize_text(text):
        doc = nlp(text)
        text = remove_punc(text)
        text = remove_tags(text)
        text = remove_stpwrds(text)
        keywords = extract_keywords(text)
        freq = Counter(keywords)
        max_freq = freq.most_common(1)[0][1]
        for i in freq.keys():
            freq[i] = freq[i] / max_freq

        sent_strength = {}
        
        for sent in doc.sents:
            for word in sent:
                if word.text in freq.keys():
                    if sent in sent_strength.keys():
                        sent_strength[sent] += freq[word.text]
                    else:
                        sent_strength[sent] = freq[word.text]

        summarized_sentences = nlargest(4, sent_strength, key=sent_strength.get)
        return summarized_sentences

    # Call the summarization function and return the result
    summary = summarize_text(text)
    return summary


with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)


class Form2(FlaskForm):
    text = StringField('Enter the text', validators=[DataRequired()])
    submit = SubmitField('Submit')


@app.route('/ai', methods=['GET', 'POST'])
def home1():
    form = Form2()
    Predictions = None
    if form.validate_on_submit():
        text = form.text.data
        Predictions = Predictionss(text)
    return render_template('ai.html', form=form, prediction=Predictions)


def remove_tags(text):
    tags = ['\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')
    return text


def remove_punc(text):
    new_text = [x for x in text if x not in string.punctuation]
    new_text = ''.join(new_text)
    return new_text


def Predictionss(text):
    text = remove_tags(text)
    text = remove_punc(text)
    pred = pipeline.predict([text])[0]
    return pred

class Form3(FlaskForm):
    text = StringField('Enter text to search', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/cb', methods=['GET', 'POST'])
def home2():
    form = Form3()
    co = cohere.Client('1rW0cpIarZTZvcThfb0CYzzdLkH2ytsME75PxcrP')

    if form.validate_on_submit():
        text = form.text.data
        response = co.generate(
            model='command-nightly',
            prompt=text,
            max_tokens=300,
            temperature=0.9,
            k=0,
            p=0.75,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        output = response.generations[0].text
        return render_template('cb.html', form=form, output=output)

    return render_template('cb.html', form=form, output=None)

class Form4(FlaskForm):
    text = StringField('Enter the text', validators=[DataRequired()])
    submit = SubmitField('Submit')

# Helper functions
def lower(text):
    return text.lower()

def removeTags(text):
    tags = ['\n\n', '\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')
    return text

def removePunct(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def removeStopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.lower() not in stop_words]
    return words

@app.route('/tp', methods=['GET', 'POST'])
def new():
    form = Form4()
    pred = None
    if form.validate_on_submit():
        text = form.text.data
        processed_text = removeStopwords(removePunct(removeTags(lower(text))))
        ptext = [processed_text]
        dictionary = corpora.Dictionary(ptext)
        corpus = [dictionary.doc2bow(doc) for doc in ptext]
        lda_model = gensim.models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
        topics = lda_model.print_topics(num_words=4)
        
        all_topic_dicts = []
        for topic in topics:
            topic_str = topic[1]
            res = topic_str.split('+')
            list1 = []
            for i in res:
                list1.append(i.split('*'))
            
            dict1 = {}
            for i in list1:
                dict1[i[1].strip().strip('"')] = float(i[0])
            
            all_topic_dicts.append(dict1)

        df1 = pd.DataFrame(list(all_topic_dicts[0].items()), columns=['Word', 'Weight'])
        df2 = pd.DataFrame(list(all_topic_dicts[1].items()), columns=['Word', 'Weight'])
        
        pred = {'Topic 1': df1.to_dict(orient='records'), 'Topic 2': df2.to_dict(orient='records')}

    return render_template('tp.html', form=form, pred=pred)



if __name__ == '__main__':
    app.run(debug=True)