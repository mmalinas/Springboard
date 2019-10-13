# predictor_app.py
import flask
from flask import request
from predictor_api import make_prediction

# Initialize the app
app = flask.Flask(__name__)
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
            make_prediction(request.args['title_in'])
        #print(x_input)
        return flask.render_template('predictor.html',
                                     title_in=x_input,
                                     prediction=predictions)
    else: 
        #For first load, request.args will be an empty ImmutableDict
        # type. If this is the case we need to pass an empty string
        # into make_prediction function so no errors are thrown.
        
        x_input, predictions = make_prediction('')
        return flask.render_template('predictor.html',
                                     title_in=x_input,
                                     prediction=predictions)

# Start the server, continuously listen to requests.
if __name__=="__main__":
    # For local development, set to True:
    app.run(debug=True)
    # For public web serving:
    #app.run(host='0.0.0.0')
    #app.run()