from flask import Flask, render_template, request
import predict_iris
import numpy as np

app = Flask(__name__)

mk=0
@app.route("/", methods = ["GET","POST"])
def hello():
    mk=0
    if request.method == "POST":
        SepalLengthCm = request.form['SepalLengthCm']
        SepalWidthCm = request.form['SepalWidthCm']
        PetalLengthCm = request.form['PetalLengthCm']
        PetalWidthCm = request.form['PetalWidthCm']
        llist = [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
        list2 = np.array(llist, dtype=float)
        marks_pred = predict_iris.flower_prediction(list2)
        mk = marks_pred

    return render_template("index.html",my_marks=mk)
'''
@app.route("/sub", methods = ['POST'])
def submit():
    # html -> .py
    if request.method == "POST":
        name = request.form["username"]

    # .py -> html
    return render_template("s1.html", n = name)
'''



if __name__ == "__main__":
    app.run(debug = True)