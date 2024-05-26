from myapp.model import ScrapedData 
from django.core.management.base import BaseCommand
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Command(BaseCommand):
    help = 'Analyzes market data from Deriv and provides trading signals based on rise/fall strategy'

    def add_arguments(self, parser):
        parser.add_argument('scraped_data_file', type=str, help='Path to the scraped data JSON file')

    def handle(self, *args, **kwargs):
        scraped_data_file = kwargs['scraped_data_file']

        # Read JSON data from file
        with open(scraped_data_file, "r") as file:
            json_data = json.load(file)

        # Convert JSON data to Pandas DataFrame
        df = pd.DataFrame(json_data)

        # Ensure 'close', 'high', 'low', and 'open' columns exist
        required_columns = ['close', 'high', 'low', 'open']
        if not all(col in df.columns for col in required_columns):
            self.stdout.write(self.style.ERROR(f"Required columns {required_columns} are missing in the provided data"))
            return

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Perform data analysis and generate signals
        analysis_result = self.perform_data_analysis(df)

        # Optionally perform machine learning training
        model, accuracy = self.train_machine_learning_model(df)

        # Display analysis result and model accuracy
        self.stdout.write(self.style.SUCCESS(f"Analysis Result:\n{analysis_result}"))
        self.stdout.write(self.style.SUCCESS(f"Model Accuracy: {accuracy}"))

    def calculate_indicators(self, df):
        # Calculate MACD and MACD Signal
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Calculate Donchian Channel
        df['Donchian_High'] = df['high'].rolling(window=20).max()
        df['Donchian_Low'] = df['low'].rolling(window=20).min()

        # Calculate CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=20).mean()
        md = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df['CCI'] = (tp - ma) / (0.015 * md)

        # Calculate Moving Averages
        df['MA_7'] = df['close'].rolling(window=7).mean()
        df['MA_12'] = df['close'].rolling(window=12).mean()
        df['MA_24'] = df['close'].rolling(window=24).mean()

        return df

    def perform_data_analysis(self, df):
        signals = []

        # Signal generation based on indicators
        for i in range(1, len(df)):
            if df['MACD'][i] > df['MACD_Signal'][i] and df['close'][i] > df['Donchian_High'][i-1] and df['CCI'][i] > 100:
                signals.append('RISE')
            elif df['MACD'][i] < df['MACD_Signal'][i] and df['close'][i] < df['Donchian_Low'][i-1] and df['CCI'][i] < -100:
                signals.append('FALL')
            else:
                signals.append('HOLD')

        df['Signal'] = signals

        # Frequency analysis for rise and fall
        rise_count = signals.count('RISE')
        fall_count = signals.count('FALL')
        total_signals = len(signals)
        rise_percentage = (rise_count / total_signals) * 100
        fall_percentage = (fall_count / total_signals) * 100

        # Plotting the frequency of rise and fall signals
        plt.figure(figsize=(10, 5))
        plt.bar(['Rise', 'Fall'], [rise_percentage, fall_percentage], color=['green', 'red'])
        plt.xlabel('Signal Type')
        plt.ylabel('Percentage')
        plt.title('Frequency of Rise and Fall Signals')
        plt.show()

        analysis_result = {
            'rise_percentage': rise_percentage,
            'fall_percentage': fall_percentage,
            'signals': signals
        }

        return analysis_result

    def train_machine_learning_model(self, df):
        # Prepare data for training - ensure 'Signal' column exists
        if 'Signal' not in df.columns:
            self.stdout.write(self.style.ERROR("'Signal' column is missing in the provided data"))
            return None, None

        # Convert signals to binary values for training (RISE=1, FALL=0)
        df['Signal'] = df['Signal'].apply(lambda x: 1 if x == 'RISE' else (0 if x == 'FALL' else None))
        df = df.dropna(subset=['Signal'])

        X = df[['MACD', 'MACD_Signal', 'Donchian_High', 'Donchian_Low', 'CCI', 'MA_7', 'MA_12', 'MA_24']]  # Feature columns
        y = df['Signal']  # Target column

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train machine learning model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Test the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return model, accuracy
