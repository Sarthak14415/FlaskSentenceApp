from pymongo import MongoClient  # ← This is the key part!
from sentence_transformers import SentenceTransformer, util
import torch
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

# Replace with your actual connection string (rotate if you shared it)
uri = "mongodb+srv://jagritjain787:Rnxsw1A40JINrv0I@cluster0.hli27ts.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
try:
    client = MongoClient(uri)
    # Ping the server to confirm connection
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB!")
except Exception as e:
    print("❌ Connection failed:")
    print(e)

print("📂 Databases in this MongoDB cluster:")
for db_name in client.list_database_names():
    print(f"• {db_name}")

db = client["test"]
collection = db["founditems"]
for doc in collection.find():
    print(doc)

collection = db["test"]

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
db = SQLAlchemy(app)

# Database model
class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'description': self.description
        }

# Create database tables
with app.app_context():
    db.create_all()

# Initialize the sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@app.route('/search', methods=['POST'])
def search_description():
    search_text = request.json.get('text')
    if not search_text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Get all data from database
    all_data = Data.query.all()
    
    if not all_data:
        return jsonify([])
    
    # Convert search text to embedding
    search_embedding = model.encode(search_text, convert_to_tensor=True)
    
    # Get all descriptions with their embeddings and calculate similarities
    best_match = None
    highest_similarity = -1
    
    for item in all_data:
        desc_embedding = model.encode(item.description, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(search_embedding, desc_embedding).item()
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = item
    
    # Return only if we found a match with similarity > 0.3
    if best_match and highest_similarity > 0.3:
        return jsonify({
            **best_match.to_dict(),
            'similarity_score': highest_similarity
        })
    
    return jsonify({'error': 'No relevant matches found'}), 404

@app.route('/add_sample_data', methods=['POST'])
def add_sample_data():
    sample_data = [
        {
            'text': 'Lost blue backpack',
            'description': 'Found a blue Nike backpack near the library. It contains some textbooks and a water bottle.'
        },
        {
            'text': 'Found keys',
            'description': 'Set of keys found in the cafeteria. Has a red keychain and about 3 keys.'
        },
        {
            'text': 'Lost student ID',
            'description': 'Found a student identification card in the parking lot. Name starts with J.'
        },
        {
            'text': 'Found laptop charger',
            'description': 'Black Dell laptop charger found in Room 201 of the Engineering building.'
        }
    ]
    
    try:
        for item in sample_data:
            new_item = Data(text=item['text'], description=item['description'])
            db.session.add(new_item)
        db.session.commit()
        return jsonify({'message': 'Sample data added successfully'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Test the search_description function with JSON data
    with app.app_context():
        test_data = {"text": "pink milton bottle"}
        with app.test_request_context(json=test_data):
            response = search_description()
            print("Test Response:", response.get_json())
    
    app.run(debug=True)