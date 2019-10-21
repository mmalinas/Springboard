# Classifying Real and Satirical News

This is the Github folder for my second capstone project, "Classifying Real and Satirical News: r/theOnion, r/nottheonion, and r/news". In this project, I used natural language processing and Naive Bayes to create a machine learning model that could distinguish between satirical and real news headlines. I achieved an F1 score of 0.72 for distinguishing between satirical news from r/theOnion and real news that appears satirical from r/nottheonion. I achieved an F1 score of 0.79 for distinguishing between r/theOnion and r/news, which posts regular news headlines.

The main data analysis for the project is contained in DataAnalysis_Final.ipynb. The rest of the files are files for the website that I created for this project using Flask and Google App Engine, which can be found at https://onion-nottheonion-app.appspot.com/. The main files that I used to create the app are main.py, which contains the code for Flask; class_def.py, which contains most of the code for the functions that I used to make the app (adapted from DataAnalysis_Final.ipynb); and predictor_api.py, which creates the function to make the predictions. The static and templates folders are for the HTML, CSS, and JavaScript for the project, for which I used a Bootstrap template.

The final report for this project with an explanation of my analysis can be found in MRM_Capstone2_FinalReport.pdf.


