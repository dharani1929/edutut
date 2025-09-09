from flask import Flask, render_template, request, session, jsonify
import os, json, secrets
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['USER_DB'] = 'users.json'
app.config['QUIZ_DB'] = 'quiz_results.json'

# ----------------------
# AI Model Initialization
# ----------------------

MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"
print(f"Loading model: {MODEL_NAME}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")
    print("Model loaded successfully.")
    
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None


def generate_ai_response(user_message):
    """Generate AI chat response using the loaded Hugging Face model."""
    if not model or not tokenizer:
        return None, "AI model is not loaded. Please check the server logs."

    messages = [
        {"role": "system", "content": "You are a helpful and responsible AI assistant that specializes in creating quizzes. You will provide responses in a structured JSON format only."},
        {"role": "user", "content": user_message}
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            thinking=False
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

        raw_output = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        # The model might include some conversational text outside the JSON.
        # We need to find and extract the valid JSON part.
        json_start = raw_output.find('[')
        json_end = raw_output.rfind(']')
        if json_start != -1 and json_end != -1:
            json_string = raw_output[json_start : json_end + 1]
            return json.loads(json_string), None
        
        return None, "Model response did not contain a valid JSON format."

    except Exception as e:
        print(f"Error during AI generation: {e}")
        return None, f"An error occurred while generating questions: {e}"


# ----------------------
# User DB Helpers (file-based)
# ----------------------
def init_db(db_file):
    if not os.path.exists(db_file):
        with open(db_file, 'w') as f:
            json.dump([], f)

def get_data(db_file):
    init_db(db_file)
    with open(db_file, 'r') as f:
        return json.load(f)

def save_data(db_file, data):
    with open(db_file, 'w') as f:
        json.dump(data, f, indent=2)

def find_user(email):
    for user in get_data(app.config['USER_DB']):
        if user['email'] == email:
            return user
    return None

def register_user(email, password):
    if find_user(email):
        return False, "Email already registered"
    users = get_data(app.config['USER_DB'])
    users.append({
        'email': email,
        'password': generate_password_hash(password),
        'created_at': datetime.now().isoformat()
    })
    save_data(app.config['USER_DB'], users)
    return True, ""

def verify_user(email, password):
    user = find_user(email)
    if not user:
        return False, "User not found"
    if not check_password_hash(user['password'], password):
        return False, "Incorrect password"
    return True, user

def save_quiz_result(email, topic, score, total):
    results = get_data(app.config['QUIZ_DB'])
    results.append({
        'user_email': email,
        'topic': topic,
        'score': score,
        'total': total,
        'date': datetime.now().isoformat()
    })
    save_data(app.config['QUIZ_DB'], results)

def get_quiz_history(email):
    results = get_data(app.config['QUIZ_DB'])
    return [r for r in results if r['user_email'] == email]

# ----------------------
# Routes
# ----------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    success, result = verify_user(email, password)
    if success:
        session['user'] = {'email': email}
        return jsonify({'status': 'success', 'user': {'email': email}})
    else:
        return jsonify({'status': 'error', 'message': result}), 401

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    success, message = register_user(email, password)
    if success:
        session['user'] = {'email': email}
        return jsonify({'status': 'success', 'user': {'email': email}})
    else:
        return jsonify({'status': 'error', 'message': message}), 400

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({'status': 'success'})

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    if 'user' not in session:
        return jsonify({"status": "error", "message": "Not logged in"}), 401
    
    data = request.json
    topic = data.get("topic", "General Knowledge")
    num_questions = data.get("num_questions", 5)

    prompt = f"Generate {num_questions} multiple-choice quiz questions about {topic}. Each question should have 4 options. Indicate the correct answer. Return the response as a valid JSON array. Do not include any text outside of the JSON. The JSON should follow this structure: [{{'question': 'Your question here', 'options': ['Option 1', 'Option 2', 'Option 3', 'Option 4'], 'answer': 2}}] The 'answer' is the zero-based index of the correct option."
    
    questions, error = generate_ai_response(prompt)
    if questions:
        return jsonify({"status": "success", "questions": questions})
    else:
        return jsonify({"status": "error", "message": error}), 500

@app.route('/save_result', methods=['POST'])
def save_result():
    if 'user' not in session:
        return jsonify({"status": "error", "message": "Not logged in"}), 401

    data = request.json
    email = session['user']['email']
    topic = data.get('topic')
    score = data.get('score')
    total = data.get('total')
    
    if not all([topic, score, total]):
        return jsonify({"status": "error", "message": "Missing quiz data"}), 400

    save_quiz_result(email, topic, score, total)
    return jsonify({"status": "success"})

@app.route('/quiz_history', methods=['GET'])
def quiz_history():
    if 'user' not in session:
        return jsonify({"status": "error", "message": "Not logged in"}), 401
    
    email = session['user']['email']
    history = get_quiz_history(email)
    return jsonify({"status": "success", "history": history})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
