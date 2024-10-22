from flask import Flask, render_template, request  
import joblib  
import numpy as np  

app = Flask(__name__)  

# Updated model paths dictionary  
model_paths = {  
    'knn': 'models/knn_model.pkl',  
    'gnb': 'models/naive_bayes_model.pkl',  
    'dtree': 'models/decision_tree_model.pkl'  
}  

# Mapping of model keys to friendly names  
classifier_names = {  
    'knn': 'Nearest Neighbor',  
    'gnb': 'Naive Bayes',  
    'dtree': 'Decision Tree'  
}  

@app.route('/')  
def index():  
    return render_template('index.html')  

@app.route('/classify', methods=['POST'])  
def classify():  
    # Get form data  
    clump_thickness = float(request.form['clump-thickness'])  
    uniformity_cell_size = float(request.form['uniformity-cell-size'])  
    uniformity_cell_shape = float(request.form['uniformity-cell-shape'])  
    marginal_adhesion = float(request.form['marginal-adhesion'])  
    single_epithelial_cell_size = float(request.form['single-epithelial-cell-size'])  
    bland_chromatin = float(request.form['bland-chromatin'])  
    normal_nucleoli = float(request.form['normal-nucleoli'])  
    mitoses = float(request.form['mitoses'])  
    classifier = request.form['classifier']  # Get selected classifier  
    friendly_name = classifier_names[classifier]  # Get friendly name  

    # Prepare the test input  
    test_set = np.array([[clump_thickness, uniformity_cell_size, uniformity_cell_shape,   
                          marginal_adhesion, single_epithelial_cell_size, bland_chromatin,   
                          normal_nucleoli, mitoses]])  

    # Load and select the model based on user input  
    model = joblib.load(model_paths[classifier])  

    # Make the prediction  
    prediction = model.predict(test_set)  
    prediction_label = ""  

    # Map the prediction to labels  
    if prediction[0] == 2:  
        prediction_label = "Benign"  
    elif prediction[0] == 4:  
        prediction_label = "Malignant"  

    # Prepare inputs for display  
    inputs = f"{clump_thickness}, {uniformity_cell_size}, {uniformity_cell_shape}, {marginal_adhesion}, {single_epithelial_cell_size}, {bland_chromatin}, {normal_nucleoli}, {mitoses}"  

    # Render the result in a template  
    return render_template('index.html', prediction=prediction_label, inputs=inputs, classifier=friendly_name)  

if __name__ == "__main__":  
    app.run(debug=True)