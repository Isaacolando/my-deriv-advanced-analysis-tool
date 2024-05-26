from flask import request, jsonify
from . import db, bcrypt
from .models import User
from . import create_app

app = create_app()

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data['email']
    name = data['name']
    password = data['password']

    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email already registered'}), 409

    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(email=email, name=name, password_hash=password_hash)

    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User created successfully'}), 201

@app.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    email = data['email']
    password = data['password']

    user = User.query.filter_by(email=email).first()

    if user and bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({'message': 'Login successful'}), 200

    return jsonify({'message': 'Invalid credentials'}), 401
