
from django.core.management.base import BaseCommand
from myapp.model import ScrapedData
import json
import pymysql
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class Command(BaseCommand):
    help = 'Analyzes the scraped data'

    def handle(self, *args, **kwargs):
        # Connect to MySQL database
        connection = pymysql.connect(
            host="localhost",
            user="wwedid",
            password="njugia",
            database="wwedid49",
            cursorclass=pymysql.cursors.DictCursor
        )

        try:
            with connection.cursor() as cursor:
                # Read scraped data from the JSON file
                with open('scraped_data.json', 'r') as file:
                    json_data = json.load(file)

                # Save scraped data to the MySQL database
                for data in json_data:
                    sql = "INSERT INTO myapp_scrapeddata (data_field) VALUES (%s)"
                    cursor.execute(sql, (data,))
                connection.commit()

                self.stdout.write(self.style.SUCCESS('Data stored successfully.'))

                # Load scraped data from the database
                scraped_data = ScrapedData.objects.all().values_list('field_name', flat=True)

                # Convert scraped data to a DataFrame
                df = pd.DataFrame(scraped_data, columns=['field_name'])

                # Perform data analysis using Pandas
                analysis_result = self.perform_data_analysis(df)

                # Perform machine learning training
                model = self.train_machine_learning_model(df)

                # Display analysis result
                self.stdout.write(str(analysis_result))

        finally:
            connection.close()

    def perform_data_analysis(self, df):
        # Perform data analysis operations using Pandas
        analysis_result = df.describe()  # Example data analysis operation
        return analysis_result

    def train_machine_learning_model(self, df):
        # Prepare data for training
        X = df.drop('target_column', axis=1)  # Adjust 'target_column' as per your model
        y = df['target_column']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train machine learning model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        return model
