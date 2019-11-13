# predictor_app.py
#website 2
import flask
from flask import redirect
from flask import request
from flask_talisman import Talisman
import predictor_api
from predictor_api import make_prediction
import pickle_util
import json

# Initialize the app
app = flask.Flask(__name__, static_url_path="/static", static_folder='static')
Talisman(app)

#app = flask.Flask(__name__, static_folder='website2/static')
# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/)
@app.route("/", methods=["GET"])
def predict():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the    
    # textbox" (value)
    #print(request.args)
    if(request.args):   
        x_input, predictions = \
            predictor_api.make_prediction(request.args['title_in'])
        #print(x_input)
        return flask.render_template('index.html',
                                     title_in=x_input,
                                     prediction=predictions)
    else: 
        #For first load, request.args will be an empty ImmutableDict
        # type. If this is the case we need to pass an empty string
        # into make_prediction function so no errors are thrown.
        x_input, predictions = predictor_api.make_prediction('')
        return flask.render_template('index.html',
                                     title_in=x_input,
                                     prediction=predictions)

@app.route("/about")
def about():
  return flask.render_template("about.html")

@app.route("/contact")
def contact():
    return flask.render_template("contact.html")

# Start the server, continuously listen to requests.
if __name__=="__main__":
    clf = pickle_util.load_clf('clf_bigrams_pickled')
    vectorizer = pickle_util.load_vectorizer('vectorizer_bigrams_pickled')
    predictor = pickle_util.load_predictor('predictor_pickled')
    app.run(debug=False)
    # For public web serving:
    #app.run(host='0.0.0.0')