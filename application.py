from flask import Flask, render_template, request
import pickle

application = Flask(__name__)
model = pickle.load(open('./models/model.pkl', 'rb'))
scaler = pickle.load(open('./models/scaler.pkl', 'rb'))

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        height = scaler.transform([[float(request.form['Height'])]])
        prediction = model.predict(height)
        return render_template('index.html', result=prediction[0])
    return render_template('index.html')

if __name__ == '__main__':
    application.run(debug=True)
