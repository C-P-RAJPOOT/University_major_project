from flask import Flask, send_file

app = Flask(__name__)

# Route to serve boxplot.png
@app.route('/boxplot')
def get_boxplot():
    return send_file("boxplot.png", mimetype='image/png')

# Route to serve confusion_matrix.png
@app.route('/confusion_matrix')
def get_confusion_matrix():
    return send_file("confusion_matrix.png", mimetype='image/png')

# Route to serve total_score_boxplot.png
@app.route('/total_score_boxplot')
def get_total_score_boxplot():
    return send_file("total_score_boxplot.png", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
