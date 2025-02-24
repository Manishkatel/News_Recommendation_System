from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import distance
from flask import Flask, request, render_template

# Initialize OpenAI client
client = OpenAI(api_key="<OPENAI_API_KEY>")

# Function to create embeddings
def create_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    response_dict = response.model_dump()
    return [data['embedding'] for data in response_dict['data']]

# News articles dataset
articles = [
    {"headline": "Economic Growth Surges in 2025", "topic": "Economy", "description": "The economy is booming with GDP growing by 5% in the first quarter."},
    {"headline": "AI is Transforming Healthcare", "topic": "Technology", "description": "AI is being used to diagnose diseases and personalize treatments."},
    {"headline": "Stock Market Hits Record High", "topic": "Finance", "description": "The stock market reached an all-time high due to investor optimism."},
    {"headline": "Climate Change: What’s Next?", "topic": "Environment", "description": "Global leaders discuss the next steps in combating climate change."},
    {"headline": "New Advances in Quantum Computing", "topic": "Technology", "description": "Scientists make breakthroughs in quantum computing algorithms."},
    {"headline": "SpaceX Launches New Mission to Mars", "topic": "Science", "description": "SpaceX successfully launches its latest Mars-bound spacecraft."},
    {"headline": "Electric Cars are the Future of Transport", "topic": "Automobile", "description": "EV adoption is increasing with new battery innovations."},
    {"headline": "Cryptocurrency Adoption is Growing", "topic": "Finance", "description": "More businesses are accepting cryptocurrency payments."},
    {"headline": "Global Trade Faces Uncertainty", "topic": "Business", "description": "Tariffs and supply chain disruptions are causing trade issues."},
    {"headline": "Renewable Energy Investments on the Rise", "topic": "Environment", "description": "Investors are putting billions into solar and wind energy."}
]

# Extract headlines and create embeddings
headline_texts = [article['headline'] for article in articles]
embeddings = create_embeddings(headline_texts)

# Store embeddings in articles
for i, article in enumerate(articles):
    article['embedding'] = embeddings[i]

# Flask Web App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_text = request.form['query']
        search_embedding = create_embeddings([search_text])[0]
        
        # Compute cosine similarity
        distances = [distance.cosine(search_embedding, article["embedding"]) for article in articles]
        
        # Get top 3 most relevant articles
        sorted_indices = np.argsort(distances)[:3]
        recommendations = [articles[i] for i in sorted_indices]
        
        return render_template('index.html', query=search_text, recommendations=recommendations)
    
    return render_template('index.html', query=None, recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)
    