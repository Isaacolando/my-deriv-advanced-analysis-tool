
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

        # Frequency/percentage analysis for rise and fall
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
    

def perfom_data_analysis(self,df):
    flattened_data=np.concatenate(df['data_field'].apply(lambda x:list(map(int, list(x)))).values)




 



