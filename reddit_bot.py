# Reddit libraries
import praw

# Machine learning libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize Reddit object to interact with site
reddit = praw.Reddit (
    client_id = 'Qcx-cc8AVCrK6A',
    client_secret = '_fV98hquZcDjHvY3Zdo8YszzGHY',
    user_agent = 'reddit_script by /u/waspswizzler'
)

# Checks that it successfully interacted with reddit
if reddit.read_only:
    print("Log: Contact with Reddit has been established")
else:
    print("Log: Contact with Reddit has failed")

# Gets top 10 submissions from subreddit
uw_submission_list = list(reddit.subreddit('uwaterloo').hot(limit=10))
print("Log: You have obtained submissions from /r/uwaterloo")

# Get a submission from subreddit
uw_submission = uw_submission_list[0]
print("Log: Top /r/uwaterloo submission is '%s'" % uw_submission.title)

# Get comments from top submissions
uw_comments = uw_submission.comments.list()
print("Log: You have obtained %d comments from /r/uwaterloo" % len(uw_comments))

# Different subreddit
ut_submission_list = list(reddit.subreddit('uoft').hot(limit=10))
print("Log: You have obtained submissions from /r/uoft")

ut_submission = ut_submission_list[3]
print("Log: Top /r/uoft submission is '%s'" % ut_submission.title)

ut_comments = ut_submission.comments.list()
print("Log: You have obtained %d comments from /r/uoft" % len(ut_comments))

# Put all comments in corpus and tag ut_comments
# 0 = /r/uwaterloo
# 1 = /r/uoft
corpus = [comment.body for comment in (uw_comments[:50] + ut_comments[:50])]
y_train = [0] * len(uw_comments[:50]) + [1] * len(ut_comments[:50])

# Vectorize corpus
vectorizer = CountVectorizer()
vectorizer.fit(corpus)
x_train = vectorizer.transform(corpus)

# Train Naive Bayes machine learning model
classifier = MultinomialNB()
classifier.fit(x_train, y_train)

# Get testing data
test_uw_comments = uw_submission_list[2].comments.list()[:10]
test_ut_comments = ut_submission_list[4].comments.list()[:10]
test_comments = test_uw_comments + test_ut_comments

# Put testing data in corpus and tag
test_corpus = [comment.body for comment in test_comments]
y_test = [0] * len(test_uw_comments) + [1] * len(test_ut_comments)

# Vectorize testing data
x_test = vectorizer.transform(test_corpus)

# Check model performance
print(classifier.predict(x_test))
print(y_test)
