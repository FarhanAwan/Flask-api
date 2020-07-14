import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import joblib
from flask import Response
from flask import send_file

app = Flask(__name__,template_folder='template')


#model = pickle.load(open('expert_model.pkl', 'rb'))
model = joblib.load('expert_model.pkl')
print(model)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload')
def upload_file():
   return render_template('upload.html')


@app.route('/predict',methods=['POST'])
def predict():    
    f = request.files['file']
    print(f.filename)
    #sTesting a single observation
    testd = pd.read_csv(f.filename)
    testd = testd.values
    labelencoder_test = LabelEncoder()
    testd[:, 0] = labelencoder_test.fit_transform(testd[:, 0])
    y_show = model.predict(testd)
    print(y_show)
    # result = ""
    if int(y_show[0][0]) == 1:
        result = "Faculty OF  Engineering"
        print("Faculty OF  Engineering")
    elif int(y_show[0][1]) == 1:
        print("Faculty Of Arts and Social Sciences")
        result = "Faculty Of Arts and Social Sciences"
    elif int(y_show[0][2]) == 1:
        print("Faculty Of Pharmacy and Pharmaceutical Sciences")
        result = "Faculty Of Pharmacy and Pharmaceutical Sciences"
    elif int(y_show[0][3]) == 1:
        print("Faculty Of Science")
        result = "Faculty Of Science"
        
    f = open("result.txt", "w")
    f.write(result)
    f.close()
    
    try:
        return render_template('result.html', something=result)
    except Exception as e:
        return str(e)
    
        
        
            
        
    
   
    


if __name__ == "__main__":
    app.run(debug=False)
    
    
#return Response(result, mimetype='text/plain')
    #return result
    #return html('Incorrect password. <a href="/">Go back?</a>')
    # return render_template('upload.html', prediction_text='Employee Salary should be {}'.format(result))
    # resp=send_file('result.txt',conditional=True)
    # return resp 
    # py = send_file('result.txt', attachment_filename='result.txt')
    # print(py)
    
    
#return render_template('upload.html', prediction_text='Employee Salary should be {}'.format(result))
    
    # try:    
    #     return send_file('result.txt', as_attachment=True,attachment_filename='result.txt')
    # except Exception as e:
    # 		return "str(e)"