
from pymongo import MongoClient
import pprint

client = MongoClient("mongodb://localhost:27017/")
db = client["admin_site"]

# Explicitly create a collection
collection_name = "data"
if collection_name not in db.list_collection_names():
    db.create_collection(collection_name)
    print(f"✅ Collection '{collection_name}' created.")
else:
    print(f"ℹ️ Collection '{collection_name}' already exists.")

logs = db["change_logs"]

with logs.watch() as stream:
    print("🔄 Listening for admin changes...")
    for change in stream:
        print("\n📝 Change Detected:")
        pprint.pprint(change["fullDocument"])
