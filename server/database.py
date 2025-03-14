from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
import os
from datetime import datetime
import pathlib
import sys
import time

# Load environment variables
load_dotenv()

# Get MongoDB URI from environment variable or use default
MONGODB_URI = os.getenv('MONGODB_URI', "mongodb+srv://gandretij20:nDOIne54TQ4xY94l@signs.wvkph.mongodb.net/videocall")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        self._connect_with_retry()

    def _connect_with_retry(self):
        retries = 0
        while retries < MAX_RETRIES:
            try:
                print(f"Attempting to connect to MongoDB (attempt {retries + 1}/{MAX_RETRIES})...")
                self.client = MongoClient(
                    MONGODB_URI,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000
                )
                # Test the connection
                self.client.admin.command('ping')
                print("Successfully connected to MongoDB!")
                
                self.db = self.client.videocall
                
                # Initialize collections
                self.rooms = self.db.rooms
                self.gestures = self.db.gestures
                self.users = self.db.users
                
                # If we get here, connection was successful
                return
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                print(f"Failed to connect to MongoDB (attempt {retries + 1}/{MAX_RETRIES}): {str(e)}")
                retries += 1
                if retries < MAX_RETRIES:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                print(f"Unexpected error connecting to MongoDB: {str(e)}")
                break
        
        print("Failed to establish MongoDB connection after all retries")

    def create_room(self, room_id, created_by):
        return self.rooms.insert_one({
            'room_id': room_id,
            'created_by': created_by,
            'created_at': datetime.now(),
            'active': True,
            'participants': [created_by],
            'gesture_history': []
        })

    def join_room(self, room_id, user_id):
        return self.rooms.update_one(
            {'room_id': room_id},
            {'$addToSet': {'participants': user_id}}
        )

    def leave_room(self, room_id, user_id):
        return self.rooms.update_one(
            {'room_id': room_id},
            {'$pull': {'participants': user_id}}
        )

    def add_gesture(self, room_id, user_id, gesture_data):
        gesture_record = {
            'room_id': room_id,
            'user_id': user_id,
            'gesture': gesture_data,
            'timestamp': datetime.now()
        }
        
        # Add to gestures collection
        self.gestures.insert_one(gesture_record)
        
        # Add to room's gesture history
        return self.rooms.update_one(
            {'room_id': room_id},
            {'$push': {
                'gesture_history': gesture_record
            }}
        )

    def get_room_gestures(self, room_id, limit=50):
        return list(self.gestures.find(
            {'room_id': room_id},
            {'_id': 0}
        ).sort('timestamp', -1).limit(limit))

    def get_active_room(self, room_id):
        return self.rooms.find_one({
            'room_id': room_id,
            'active': True
        })

# Create global database instance
db = Database() 