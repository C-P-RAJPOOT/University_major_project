import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_random_forest_classifier(csv_file='StudentsPerformance.csv', test_size=0.2, random_state=42):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    df.columns = ['gender', 'race', 'parent_education', 'lunch', 'course', 'math', 'reading', 'writing']

    education_translation = {'some high school': 0, 'high school': 1, 'some college': 2, "associate's degree": 3,
                             "bachelor's degree": 4, "master's degree": 5}
    df['parent_education'] = df['parent_education'].map(education_translation)

    course_to_number = {'none': 0, 'completed': 1}
    df['course'] = df['course'].map(course_to_number)

    lunch_to_number = {'free/reduced': 0, 'standard': 1}
    df['lunch'] = df['lunch'].map(lunch_to_number)

    df['total_score'] = df['math'] + df['reading'] + df['writing']

    percentiles = df['total_score'].describe(percentiles=np.arange(0, 1, 0.2))
    quintiles = percentiles[['0%', '20%', '40%', '60%', '80%', 'max']]
    labels = [1, 2, 3, 4, 5]
    df['level'] = pd.cut(df['total_score'], bins=quintiles, labels=labels, include_lowest=True)
    df['level'] = df['level'].astype(int)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[df['math'], df['reading'], df['writing']])
    plt.xticks([0, 1, 2], ['math', 'reading', 'writing'])
    plt.xlabel('Test')
    plt.ylabel('Score')
    plt.title('Boxplot for each test')
    boxplot_path = os.path.join(current_dir, 'boxplot.png')
    plt.savefig(boxplot_path)
    plt.close()

    plt.figure(figsize=(6, 6))
    sns.boxplot(data=df['total_score'])
    plt.xticks([0], ['total_score'])
    plt.title('Boxplot for total score')
    total_score_boxplot_path = os.path.join(current_dir, 'total_score_boxplot.png')
    plt.savefig(total_score_boxplot_path)
    plt.close()

    df['gender'] = pd.Categorical(df['gender']).codes
    df['race'] = pd.Categorical(df['race']).codes
    df['level'] = pd.Categorical(df['level']).codes

    X = df[['gender', 'race', 'parent_education', 'lunch', 'course', 'math', 'reading', 'writing']]
    y = df['level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=df['level'].unique(), yticklabels=df['level'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    confusion_matrix_path = os.path.join(current_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()

    return boxplot_path, total_score_boxplot_path, confusion_matrix_path

# Example usage:
run_random_forest_classifier()
