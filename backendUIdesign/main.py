from flask import Flask, request , jsonify
app= (__name__)

@app.route("/")
def request():
    return "request"

@app.route("/", methods=['POST'])
def request():
    if request.method == 'POST':
        data = request.get_json()
   


if __name__ == '__main__':
    app.run(debug=True)
