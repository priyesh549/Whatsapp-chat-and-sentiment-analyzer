from collections import Counter

import emoji
import pandas as pd
from urlextract import URLExtract
import re

extractor = URLExtract()

from wordcloud import WordCloud, STOPWORDS

from translate import Translator

from lib2to3.pgen2.pgen import DFAState
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch number of messages
    num_messages = df.shape[0]

    # fetch number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>'].shape[0]

    # fetch urls
    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='black')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains('<Media omitted>|<media omitted>|<omitted>')]
    
    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words and not re.match(r'^\W+$', word):
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    most_common_df = most_common_df.rename(columns={0: 'Words', 1: 'Occurrences'})
    return most_common_df


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.distinct_emoji_list(message)])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    emoji_df = emoji_df.rename(columns={0: 'Emojis', 1: 'Occurences'})

    return emoji_df

# monthly timeline
def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    activity_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return activity_heatmap


#sentiment analysis using vader
def sentiment(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']


    posarr = []
    negarr = []
    neuarr = []
    overall = []
    res = []

    for message in temp['message']:
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(message)

        negarr.append(sentiment_dict['neg'] * 100)
        neuarr.append(sentiment_dict['neu'] * 100)
        posarr.append(sentiment_dict['pos'] * 100)
        overall.append(sentiment_dict['compound'] * 100)

        # decide sentiment as positive, negative and neutral
        if sentiment_dict['compound'] >= 0.05:
            res.append("Positive")

        elif sentiment_dict['compound'] <= - 0.05:
            res.append("Negative")

        else:
            res.append("Neutral")

    dict = {'Message': temp['message'],
            'Positive': posarr,
            'Negative': negarr,
            'Neutral': neuarr,
            'Overall': overall,
            'result': res
            }
    most_common_df = pd.DataFrame(dict)
    return most_common_df


def conclusion(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    tot = 0
    totMessages = 0
    res = ""

    for message in temp['message']:
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(message)
        if sentiment_dict['compound'] != 0:
            totMessages = totMessages + 1
            tot = (tot + (sentiment_dict['compound']))

    tot = (tot / totMessages)

    # decide sentiment as positive, negative and neutral
    if tot >= 0.05:
        res = "Positive"

    elif tot <= - 0.05:
        res = "Negative"

    else:
        res = "Neutral"

    dict = {
        'conclusion': [res],
        'Overall Score': [tot * 100]
    }
    most_common_df = pd.DataFrame(dict)
    return most_common_df







