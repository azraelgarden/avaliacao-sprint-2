from pymongo import MongoClient
import pandas as pd
from pprint import pprint

class Database:
    
    def __init__(self, db_name, collection_name):
        self._db = db_name
        self._collection = collection_name

        connection = f"mongodb+srv://admin:admin@cluster0.uz4j8.mongodb.net/{self._db}?retryWrites=true&w=majority"
        self.client = MongoClient(connection)
        
        database = self.client[self._db]
        self._collection = database[self._collection]


    def insert_one(self, data):
        self._collection.insert_one(data)

    def insert_many(self, data):
        self._collection.insert_many(data)

    def show(self):
        for data in self._collection.find():
            pprint(data)

    