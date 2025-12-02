import pymongo
from pymongo import MongoClient


client = MongoClient('mongodb://localhost:27018/', username='root', password='ALADIN2025')
db = client['orion']

print(db.list_collection_names())