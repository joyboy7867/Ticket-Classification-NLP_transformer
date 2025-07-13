import pandas as pd
import numpy as np
import pickle
from flask import Flask,request,render_template
app=Flask(__name__)
model=pickle.load(open('models/Logisticmodel.pkl','rb'))
transformer=pickle.load(open('models/transformer.pkl','rb'))

@app.route("/predict",methods=["POST","GET"])
def prediction():
    if request.method=="POST":
        String=request.form.get("ticket")
        data=[String]
        vector_data=transformer.encode(data)
        predicted_data=model.predict(vector_data)
        return render_template("index.html",predictions=predicted_data[0])
    return "Use POST to send ticket data"
@app.route("/",methods=["POST","GET"])
def rend():
    return render_template("index.html")

if __name__ =="__main__":
    app.run(host="0.0.0.0")