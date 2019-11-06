# Classifying Real and Satirical News:  r/theOnion, r/nottheonion, and r/news

This is the code and reports for my second capstone project, "Classifying Real and Satirical News: r/theOnion, r/nottheonion, and r/news". In this project, I created a Naive Bayes model using Reddit posts that can predict whether a headline is from the Onion or not. My model achieved an F1 score of 0.73 for Onion vs. nottheonion headlines, and an F1 score of 0.79 for Onion vs. regular news. The code for my model is in "MRM_DataExploration.ipynb". The data I used (accessing the Reddit API and saving the output as a CSV) is in "all_posts_reddit_onionandnotonion_2.csv". My final report is in "MRM_Capstone2_FinalReport".

I also developed my r/theOnion vs. r/nottheonion model with bigrams into a web-based application using Flask and Google Cloud App Engine. You can find it [here](https://onion-nottheonion-app.appspot.com/). The code for the website is in the "website-final" folder.
