from django.core.management.base import BaseCommand
import json
from myapp.model import ScrapedData 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Command(BaseCommand):
    help = 'Analyzes Volatility market data from Deriv and provides trading signals'

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

        # Perform data analysis and generate signals
        analysis_result = self.perform_data_analysis(df)

        # Perform machine learning training
        model, accuracy = self.train_machine_learning_model(df)

        # Display analysis result and model accuracy
        self.stdout.write(self.style.SUCCESS(f"Analysis Result:\n{analysis_result}"))
        self.stdout.write(self.style.SUCCESS(f"Model Accuracy: {accuracy}"))

    def perform_data_analysis(self, df):
        # Flatten the 'data_field' column which is assumed to contain sequences of digits
        flattened_data = np.concatenate(df['data_field'].apply(lambda x: list(map(int, list(x)))).values)

        # Count the frequency of each digit
        digit_counts = Counter(flattened_data)

        # Sort digits based on their frequency
        sorted_digits = sorted(digit_counts.items(), key=lambda x: x[1], reverse=True)

        # Extract the most and least frequent digits
        most_frequent_digit = sorted_digits[0][0]
        least_frequent_digit = sorted_digits[-1][0]

        signals = []

        # Strategy for Under 6
        if most_frequent_digit == 8 and least_frequent_digit == 2:
            signals.append("Signal for Under 6: Most frequent is 8, least frequent is 2")

        # Strategy for Under 7
        if most_frequent_digit in [8, 9]:
            signals.append("Signal for Under 7: Tick points to 8 or 9")

        # Strategy for Under 6 again
        if most_frequent_digit in [8, 7, 9]:
            signals.append("Signal for Under 6: Tick points to 8, 7, or 9")

        # Strategy for Under 5
        if most_frequent_digit in [6, 7, 8, 9]:
            signals.append("Signal for Under 5: Tick points to 6, 7, 8, or 9")

        # Plotting the digit frequencies
        digits, counts = zip(*sorted_digits)
        plt.figure(figsize=(10, 5))
        plt.bar(digits, counts, color='blue')
        plt.xlabel('Digits')
        plt.ylabel('Frequency')
        plt.title('Digit Frequency Analysis')
        plt.show()

        analysis_result = {
            'most_frequent_digit': most_frequent_digit,
            'least_frequent_digit': least_frequent_digit,
            'signals': signals
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
