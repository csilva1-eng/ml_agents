import os, requests
from dotenv import load_dotenv
load_dotenv()

# what ollama.com offers via the API
r = requests.get("https://ollama.com/api/tags",
                 headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"})
print("CLOUD:", [m["name"] for m in r.json()["models"]])

# what your local server offers
try:
    print("LOCAL:", [m["name"] for m in requests.get("http://localhost:11434/api/tags").json()["models"]])
except Exception as e:
    print("local server not reachable:", e)