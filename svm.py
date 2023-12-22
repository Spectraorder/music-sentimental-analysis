import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("musical_traits_truth.csv")

# Assuming you have a column named 'Emotion' with labels 0 or 1
X = df.drop("Emotion", axis=1)
y = df["Emotion"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC()

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Add the predictions to the DataFrame
X_test_with_predictions = X_test.copy()
X_test_with_predictions["Predicted_Emotion"] = y_pred

# Save the results to a new CSV file
result_df = pd.DataFrame({
    "Tempo": X_test_with_predictions["Tempo"],
    "Spectral Centroid": X_test_with_predictions["Spectral Centroid"],
    "Spectral Bandwidth": X_test_with_predictions["Spectral Bandwidth"],
    "Chroma Features": X_test_with_predictions["Chroma Features"],
    "Zero Crossing Rate": X_test_with_predictions["Zero Crossing Rate"],
    "True_Emotion": y_test,
    "Predicted_Emotion": y_pred
})

result_df.to_csv("svm_results.csv", index=False)
