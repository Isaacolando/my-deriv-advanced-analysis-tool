
from django.db import models
from django.db.models import Count
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

