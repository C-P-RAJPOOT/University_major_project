import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_randomforest():
    # Load the data
    df = pd.read_csv("StudentsPerformance.csv")

    # Renaming columns
    df.columns = ['gender', 'race', 'parent_education', 'lunch', 'course', 'math', 'reading', 'writing']

    # Mapping categorical variables to numerical values
    education_translation = {'some high school': 0, 'high school': 1, 'some college': 2, "associate's degree": 3,
                             "bachelor's degree": 4, "master's degree": 5}
    df['parent_education'] = df['parent_education'].map(education_translation)
    course_to_number = {'none': 0, 'completed': 1}
    df['course'] = df['course'].map(course_to_number)
    lunch_to_number = {'free/reduced': 0, 'standard': 1}
    df['lunch'] = df['lunch'].map(lunch_to_number)

    # Creating a new feature 'total_score'
    df['total_score'] = df['math'] + df['reading'] + df['writing']

    # Creating 'level' column
    percentiles = df['total_score'].describe(percentiles=np.arange(0, 1, 0.2))
    quintiles = percentiles[['0%', '20%', '40%', '60%', '80%', 'max']]
    labels = [1, 2, 3, 4, 5]
    df['level'] = pd.cut(df['total_score'], bins=quintiles, labels=labels, include_lowest=True)
    df['level'] = df['level'].astype(int)

    # Boxplot for each test
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[df['math'], df['reading'], df['writing']])
    plt.xticks([0, 1, 2], ['math', 'reading', 'writing'])
    plt.xlabel('Test')
    plt.ylabel('Score')
    plt.title('Boxplot for each test')
    plt.savefig('Boxplot_for_each_test.svg')
    #plt.show()

    # Boxplot for total score
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=df['total_score'])
    plt.xticks([0], ['total_score'])
    plt.title('Boxplot for total score')
    plt.savefig('Boxplot_for_total_score.svg')
    #plt.show()

    # Mapping 'gender', 'race', and 'level' to numerical values
    df['gender'] = pd.Categorical(df['gender']).codes
    df['race'] = pd.Categorical(df['race']).codes
    df['level'] = pd.Categorical(df['level']).codes

    # Features (X) and Target Variable (y)
    X = df[['gender', 'race', 'parent_education', 'lunch', 'course', 'math', 'reading', 'writing']]
    y = df['level']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Training the classifier on the training data
    rf_classifier.fit(X_train, y_train)

    # Making predictions on the testing data
    y_pred = rf_classifier.predict(X_test)

    # Evaluating the performance of the Random Forest Classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Displaying additional evaluation metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Displaying the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=df['level'].unique(),
                yticklabels=df['level'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_plot.svg')
    #plt.show()

    # Close the plot to prevent displaying it in the console
    plt.close()
    def read_svg_file(file_path):
        try:
            with open(file_path, 'r') as file:
                svg_data = file.read()
            return svg_data
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None

# Example usage:
    file_path = 'Boxplot_for_each_test.svg'
    file_path1 = 'Boxplot_for_total_score.svg'
    file_path2 = 'confusion_matrix_plot.svg'
    svg_data = read_svg_file(file_path)
    svg_data1 = read_svg_file(file_path1)
    svg_data2 = read_svg_file(file_path2)
#if svg_data:
    #print("SVG file contents:")
    #print(svg_data)



    #return accuracy, classification_report(y_test, y_pred), conf_matrix
    return svg_data,svg_data1,svg_data2
    

#run_randomforest()
     
