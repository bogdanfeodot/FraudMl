import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('models/MLP_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    print("#############################")

    features = [x for x in request.form.values()]

    print(features)

    #features = [12,'CASH_OUT',3000,"c1921",6000,3000,"m123",500,3500]

    def data_manipulation(x):

        col_names = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
        col_vals = x
        col_dict = dict(zip(col_names, col_vals))
        data = pd.DataFrame(col_dict, index =[0])

        data_new = data.copy()

        data_new = data_new[(data_new["type"] == "CASH_OUT") | (data_new["type"] == "TRANSFER")]

        data_new["errorBalanceOrg"] = data_new.newbalanceOrig + data_new.amount - data_new.oldbalanceOrg
        data_new["errorBalanceDest"] = data_new.oldbalanceDest + data_new.amount - data_new.newbalanceDest

        names = ["nameOrig","nameDest"]
        data_new = data_new.drop(names,1)

        # adding feature HourOfDay 
        data_new["HourOfDay"] = np.nan # initializing feature column
        data_new.HourOfDay = data_new.step % 24

        data_new["type_CASH_OUT"] = 0
        data_new["type_TRANSFER"] = 0

        data_new.loc[data_new['type'] == 'CASH_OUT', 'type_CASH_OUT'] = 1
        data_new.loc[data_new['type'] == 'TRANSFER', 'type_TRANSFER'] = 1

        data_new = data_new.drop("type", 1)
        data_new = data_new.apply(pd.to_numeric)

        relevant_features = data_new.to_numpy()

        return(relevant_features)

    #x = [12,'CASH_OUT',3000,"c1921",6000,3000,"m123",500,3500]

    #features = 
    
    int_features = data_manipulation(features) #[float(x) for x in data_manipulation(features)]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    '''
    return render_template('index.html', prediction_text='This transaction is:$ {}'.format(output))
    #render_template('index.html', prediction_text = 'Features list :$ {}'.format(features)) 
    #render_template('index.html', prediction_text='This transaction is:$ {}'.format(output))
    

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)