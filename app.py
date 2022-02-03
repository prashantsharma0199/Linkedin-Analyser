from flask import Flask, render_template, request, send_file
import senti
import senti_analysis

app= Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        usern = (request.form['username'])
        pswd = (request.form['password'])
        usery = (request.form['user'])
        filey = (request.form['filename'])
        senti.findData(usery, filey, usern, pswd)
        senti_analysis.analysis(usery, filey)
        return render_template('result.html', prediction=usery)

@app.route('/download', methods=['POST'])
def download():
    if request.method == 'POST':
	    filename= (request.form['us'])
	    return send_file('static\\report\\'+filename+'\\Report-'+filename+'.txt', as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
