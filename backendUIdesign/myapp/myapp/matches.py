from django.core.management.base import BaseCommand
from myapp.model import ScrapedData  # Ensure this is the correct path to your model
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Command(BaseCommand):
    help = 'Analyzes the scraped data and trains a machine learning model'

    def add_arguments(self, parser):
        parser.add_argument('scraped_data_file', type=str, help='Path to the scraped data JSON file')

    def handle(self, *args, **kwargs):
        scraped_data_file = kwargs['scraped_data_file']

        # Read JSON data from file
        with open(scraped_data_file, "r") as file:
            json_data = json.load(file)

        # Convert JSON data to Pandas DataFrame
        df = pd.DataFrame(json_data)

        # Ensure 'data_field' column exists
        if 'data_field' not in df.columns:
            self.stdout.write(self.style.ERROR("'data_field' column is missing in the provided data"))
            return

        # Perform data analysis
        analysis_result = self.perform_data_analysis(df)

        # Perform machine learning training
        model, accuracy = self.train_machine_learning_model(df)

        # Display analysis result and model accuracy
        self.stdout.write(self.style.SUCCESS(f"Analysis Result:\n{analysis_result}"))
        self.stdout.write(self.style.SUCCESS(f"Model Accuracy: {accuracy}"))

    def perform_data_analysis(self, df):
        # Flatten the 'data_field' column which is assumed to contain sequences of digits
        flattened_data = np.concatenate(df['data_field'].values)

        # Count the frequency of each digit
        digit_counts = Counter(flattened_data)

        # Sort digits based on their frequency
        sorted_digits = sorted(digit_counts.items(), key=lambda x: x[1], reverse=True)

        # Extract the 3rd and 7th most frequent digits
        third_digit = sorted_digits[2][0]
        seventh_digit = sorted_digits[6][0]

        # Generate time-series data for the identified digits (for simplicity, using random data here)
        time_series_length = 100  # Adjust as needed
        third_digit_counts = np.random.randint(50, 100, size=time_series_length)
        seventh_digit_counts = np.random.randint(30, 80, size=time_series_length)

        # Calculate percentage increase
        third_digit_percentage_increase = np.diff(third_digit_counts) / third_digit_counts[:-1] * 100
        seventh_digit_percentage_increase = np.diff(seventh_digit_counts) / seventh_digit_counts[:-1] * 100

        # Calculate the average percentage increase
        avg_third_increase = np.mean(third_digit_percentage_increase)
        avg_seventh_increase = np.mean(seventh_digit_percentage_increase)

        # Predict based on the higher average increase
        prediction = third_digit if avg_third_increase > avg_seventh_increase else seventh_digit

        # Plot the digit frequencies and trends
        plt.figure(figsize=(12, 6))
        plt.plot(range(time_series_length), third_digit_counts, label=f"Digit {third_digit} Counts")
        plt.plot(range(time_series_length), seventh_digit_counts, label=f"Digit {seventh_digit} Counts")
        plt.plot(range(1, time_series_length), third_digit_percentage_increase, label=f"Digit {third_digit} % Increase", linestyle='--')
        plt.plot(range(1, time_series_length), seventh_digit_percentage_increase, label=f"Digit {seventh_digit} % Increase", linestyle='--')
        plt.title("Digit Counts and Percentage Increase Over Time")
        plt.xlabel("Time")
        plt.ylabel("Counts / Percentage Increase")
        plt.legend()
        plt.show()

        analysis_result = {
            'third_digit': third_digit,
            'seventh_digit': seventh_digit,
            'avg_third_increase': avg_third_increase,
            'avg_seventh_increase': avg_seventh_increase,
            'prediction': prediction
        }

        return analysis_result

    def train_machine_learning_model(self, df):
        # Prepare data for training - adjust according to your dataset structure
        if 'target_column' not in df.columns:
            self.stdout.write(self.style.ERROR("'target_column' is missing in the provided data"))
            return None, None

        X = df.drop('target_column', axis=1)  # Feature columns
        y = df['target_column']  # Target column

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train machine learning model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Test the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return model, accuracy
