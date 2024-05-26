from django.apps import AppConfig

class MyAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myapp'
from flask import Flask, render_template
from flask_socketio import SocketIO, send
from flask import Flask, jsonify
          
import json
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(message):
    with open('path/to/analysis_results.json') as file:
        data = json.load(file)
    send(data)

if __name__ == '__main__':
    socketio.run(app, debug=True)

app = Flask(__name__)

@app.route('/api/analysis-results', methods=['GET'])
def get_analysis_results():
    with open('path/to/analysis_results.json') as file:
        data = json.load(file)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)      