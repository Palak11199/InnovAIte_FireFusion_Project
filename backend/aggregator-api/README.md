# Aggregator API

# Setup

1. Create a virtual environment and activate it:
python3 -m venv venv
source venv/bin/activate

2. Install dependencies:
pip install fastapi uvicorn python-dotenv

3. Create a `.env` file with your API key:

API_KEY=your-secret-key-here


# Running

uvicorn app.main:app --reload