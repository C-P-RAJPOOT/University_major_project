from flask import Flask, jsonify
import w_kmeans1 as km
import w_randomForest2 as rf
import data1 as da

app = Flask(__name__)
@app.route('/kmeans', methods=['GET'])
def run_kmeans_api():

    da.data()

    kmeans = km.run_kmeans()

    # Return the centroids and labels as a JSON response
    response = {
        'kmeans_svg' : kmeans
        
        #'centroids': kmeans.centroids.tolist(),
        #'labels': labels.tolist()
    }
    return jsonify(response)

@app.route('/randomforest', methods=['GET'])
def run_randomforest_api():

    svg,svg1,svg2 = rf.run_randomforest()
    #accuracy, classification_report_str, confusion_matrix = rf.run_randomforest()


    # Return the accuracy, classification report, and confusion matrix as a JSON response
    response = {

        'svg':svg,
        'svg1':svg1,
        'svg2':svg2,
            
            
            

        #'accuracy': accuracy,
        #'classification_report': classification_report_str,
        #'confusion_matrix': confusion_matrix.tolist()



    }
    return jsonify(response)

@app.route('/', methods=['GET'])
def home():
    return '''
    <h1>StudentsPerformance API</h1>
    <p>This API provides access to the KMeans clustering and Random Forest Classifier modules.</p>
    <h2>Endpoints:</h2>
    <ul>
        <li><a href="/kmeans">/kmeans</a>: Run the KMeans clustering algorithm.</li>
        <li><a href="/randomforest">/randomforest</a>: Run the Random Forest Classifier algorithm.</li>
    </ul>
    '''

if __name__ == '__main__':
    app.run(debug=True)