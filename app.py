from src.TicketPricespredctionofFlight.pipelines.prediction_pipeline import custom_data,model_prediction
from flask import Flask,request,render_template
import numpy as np
app=Flask(__name__,template_folder="template")
@app.route('/')
def home_page():
    return render_template("form.html")
@app.route('/predict',methods=["POST"])
def pred_page():   
        get_data=custom_data(airline=request.form.get('airline'),
                             source_city=request.form.get('source_city'),
                             departure_time=request.form.get('departure_time'),
                             stops=request.form.get('stops'),
                             arrival_time=request.form.get('arrival_time'),
                             destination_city=request.form.get('destination_city'),
                             classes=request.form.get('classes'),
                             duration=float(request.form.get('duration')),
                             days_left= float(request.form.get('days_left')))
        
        final_data=get_data.get_data_as_dataframe()
        pred=model_prediction()
        x=pred.model_pred_initiate(final_data)
        return render_template("result.html",x=x)

if __name__=="__main__":
 app.run()
