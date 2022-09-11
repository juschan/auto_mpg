import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify

app = Flask('myApp')
model = pickle.load(open('auto_mpg_lr_model.pkl', 'rb'))

# Show a form to the user
@app.route('/form')
def form():
    # use flask's render_template function to display an html page
    return render_template('form.html')


# Accept the form submission and calculate mpg
@app.route('/submit')
def make_predictions():
    # load in the form data from the incoming request
    user_input = request.args

    # manipulate data into a format that we pass to our model
    data = np.array([
        int(user_input['Cylinders']),
        int(user_input['Displacement']),
        float(user_input['Horsepower']),
        int(user_input['Weight']),
        int(user_input['Acceleration'])
    ]).reshape(1, -1)
    
    # make predictions
    prediction = model.predict(data)[0]

    # return the results template with our prediction value filled in
    return render_template('results.html', prediction=round(prediction, 2))

    #return jsonify({'data' : data})


if __name__ == '__main__':
    app.run(debug=True)