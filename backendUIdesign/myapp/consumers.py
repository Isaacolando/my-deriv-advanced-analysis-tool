import json
from channels.generic.websocket import WebsocketConsumer

class AnalysisConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        data = json.loads(text_data)
        # Perform analysis or fetch latest results
        with open('path/to/analysis_results.json') as file:
            analysis_data = json.load(file)
        self.send(text_data=json.dumps(analysis_data))