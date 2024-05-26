from django.core.management.base import BaseCommand
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Command(BaseCommand):
    help = 'Analyzes Volatility 75 (1s) market data from Deriv and provides trading signals based on even/odd strategy'

    def add_arguments(self, parser):
        parser.add_argument('scraped_data_file', type=str, help='Path to the scraped data JSON file')

    def handle(self, *args, **kwargs):
        scraped_data_file = kwargs['scraped_data_file']

        # Read JSON data from file
        with open(scraped_data_file, "r") as file:
            json_data = json.load(file)

        # Convert JSON data to Pandas DataFrame
        df = pd.DataFrame(json_data)

        # Ensure 'data_field' column and price columns exist
        if 'data_field' not in df.columns or 'close' not in df.columns:
            self.stdout.write(self.style.ERROR("'data_field' or 'close' column is missing in the provided data"))
            return

        # Calculate moving averages
        df['MA_100'] = df['close'].ewm(span=100, adjust=False).mean()
        df['MA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['MA_10'] = df['close'].ewm(span=10, adjust=False).mean()

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

        # Calculate even and odd digit counts
        even_count = sum([count for digit, count in digit_counts.items() if digit % 2 == 0])
        odd_count = sum([count for digit, count in digit_counts.items() if digit % 2 != 0])

        # Signal generation based on moving averages and candlestick colors
        df['signal'] = np.where(
            (df['MA_100'] > df['close']) & (df['MA_20'] > df['close']) & (df['MA_10'] > df['close']) & (df['close'] < df['open']),
            'EVEN', 
            np.where(
                (df['MA_100'] < df['close']) & (df['MA_20'] < df['close']) & (df['MA_10'] < df['close']) & (df['close'] > df['open']),
                'ODD', 
                None
            )
        )

        # Consecutive ticks criteria
        df['consecutive_even'] = df['data_field'].apply(lambda x: all(int(d) % 2 == 0 for d in x))
        df['consecutive_odd'] = df['data_field'].apply(lambda x: all(int(d) % 2 != 0 for d in x))

        df['tick_signal'] = np.where(
            df['consecutive_odd'], 'EVEN', 
            np.where(df['consecutive_even'], 'ODD', None)
        )

        # Combine signals
        df['combined_signal'] = df.apply(lambda row: row['signal'] if row['signal'] else row['tick_signal'], axis=1)

        # Plotting the frequency of even and odd digits
        plt.figure(figsize=(10, 5))
        plt.bar(['Even', 'Odd'], [even_count, odd_count], color=['blue', 'red'])
        plt.xlabel('Digit Type')
        plt.ylabel('Frequency')
        plt.title('Frequency of Even and Odd Digits')
        plt.show()

        # Plotting individual digit frequencies
        digits, counts = zip(*sorted(digit_counts.items()))
        plt.figure(figsize=(10, 5))
        plt.bar(digits, counts, color='blue')
        plt.xlabel('Digits')
        plt.ylabel('Frequency')
        plt.title('Digit Frequency Analysis')
        plt.show()

        analysis_result = {
            'even_count': even_count,
            'odd_count': odd_count,
            'signals': df['combined_signal'].dropna().tolist()
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
