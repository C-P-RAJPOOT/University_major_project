
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend

from flask import Flask, send_file
from kmeans6 import run_kmeans_clustering
from randomforest5 import run_random_forest_classifier

app = Flask(__name__)

@app.route('/kmeans_image')
def get_kmeans_image():
    # Run KMeans and get the image path
    image_path = run_kmeans_clustering()
    # Return the image file
    return send_file(image_path, mimetype='image/png')

@app.route('/random_forest_outputs')
def get_random_forest_outputs():
    # Run Random Forest and get the paths to all output images
    boxplot_path, total_score_boxplot_path, confusion_matrix_path = run_random_forest_classifier()
    # Return the paths to all output images
    return {
        #'boxplot': boxplot_path,
        #'total_score_boxplot': total_score_boxplot_path,
        #'confusion_matrix': confusion_matrix_path
    }

if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # Run Flask in multi-threaded mode
