from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime
import pathlib

# Set MongoDB URI directly for testing
MONGODB_URI = "mongodb+srv://gandretij20:nDOIne54TQ4xY94l@signs.wvkph.mongodb.net/videocall"

class Database:
    def __init__(self):
        try:
            print(f"Attempting to connect to MongoDB...")
            self.client = MongoClient(MONGODB_URI)
            # Test the connection
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB!")
            
            self.db = self.client.videocall
            
            # Collections
            self.rooms = self.db.rooms
            self.gestures = self.db.gestures
            self.users = self.db.users
            
        except Exception as e:
            print(f"Error connecting to MongoDB: {str(e)}")
            raise

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