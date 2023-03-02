from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/cropfire')
def cropfire():
    return render_template("cropfire.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('cropfire.html',pred='Your Crop is in Danger.\nProbability of fire occuring is {}'.format(output),b="Something need to do to save the crop")
    else:
        return render_template('cropfire.html',pred='Your Crop is safe.\n Probability of fire occuring is {}'.format(output),b="Your cropt is Safe for now")


if __name__ == '__main__':
    app.run(debug=True)
