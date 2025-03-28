from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
import os
from datetime import datetime
import pathlib
import sys
import time
import json
import threading

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
        self.connected = False
        self.in_memory_rooms = {}
        self.in_memory_gestures = {}
        self._connect_with_retry()
        
        # If MongoDB connection fails, we'll use in-memory storage
        if not self.connected:
            print("Using in-memory storage as fallback")

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
                self.connected = True
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
        self.connected = False

    def create_room(self, room_id, created_by):
        if self.connected:
            try:
                return self.rooms.insert_one({
                    'room_id': room_id,
                    'created_by': created_by,
                    'created_at': datetime.now(),
                    'active': True,
                    'participants': [created_by],
                    'gesture_history': []
                })
            except Exception as e:
                print(f"MongoDB error in create_room: {e}")
                # Fall back to in-memory storage
        
        # In-memory fallback
        self.in_memory_rooms[room_id] = {
            'room_id': room_id,
            'created_by': created_by,
            'created_at': datetime.now().isoformat(),
            'active': True,
            'participants': [created_by],
            'gesture_history': []
        }
        print(f"Room {room_id} created in memory")
        return True

    def join_room(self, room_id, user_id):
        if self.connected:
            try:
                return self.rooms.update_one(
                    {'room_id': room_id},
                    {'$addToSet': {'participants': user_id}}
                )
            except Exception as e:
                print(f"MongoDB error in join_room: {e}")
        
        # In-memory fallback
        if room_id in self.in_memory_rooms:
            if user_id not in self.in_memory_rooms[room_id]['participants']:
                self.in_memory_rooms[room_id]['participants'].append(user_id)
            return True
        return False

    def leave_room(self, room_id, user_id):
        if self.connected:
            try:
                return self.rooms.update_one(
                    {'room_id': room_id},
                    {'$pull': {'participants': user_id}}
                )
            except Exception as e:
                print(f"MongoDB error in leave_room: {e}")
        
        # In-memory fallback
        if room_id in self.in_memory_rooms:
            if user_id in self.in_memory_rooms[room_id]['participants']:
                self.in_memory_rooms[room_id]['participants'].remove(user_id)
            return True
        return False

    def add_gesture(self, room_id, user_id, gesture_data):
        gesture_record = {
            'room_id': room_id,
            'user_id': user_id,
            'gesture': gesture_data,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.connected:
            try:
                # Add to gestures collection
                self.gestures.insert_one(gesture_record)
                
                # Add to room's gesture history
                return self.rooms.update_one(
                    {'room_id': room_id},
                    {'$push': {
                        'gesture_history': gesture_record
                    }}
                )
            except Exception as e:
                print(f"MongoDB error in add_gesture: {e}")
        
        # In-memory fallback
        if room_id not in self.in_memory_gestures:
            self.in_memory_gestures[room_id] = []
        
        self.in_memory_gestures[room_id].append(gesture_record)
        
        if room_id in self.in_memory_rooms:
            if 'gesture_history' not in self.in_memory_rooms[room_id]:
                self.in_memory_rooms[room_id]['gesture_history'] = []
            self.in_memory_rooms[room_id]['gesture_history'].append(gesture_record)
        
        return True

    def get_room_gestures(self, room_id, limit=50):
        if self.connected:
            try:
                return list(self.gestures.find(
                    {'room_id': room_id},
                    {'_id': 0}
                ).sort('timestamp', -1).limit(limit))
            except Exception as e:
                print(f"MongoDB error in get_room_gestures: {e}")
        
        # In-memory fallback
        if room_id in self.in_memory_gestures:
            # Sort by timestamp (newest first) and limit
            sorted_gestures = sorted(
                self.in_memory_gestures[room_id],
                key=lambda x: x['timestamp'],
                reverse=True
            )
            return sorted_gestures[:limit]
        return []

    def get_active_room(self, room_id):
        if self.connected:
            try:
                return self.rooms.find_one({
                    'room_id': room_id,
                    'active': True
                })
            except Exception as e:
                print(f"MongoDB error in get_active_room: {e}")
        
        # In-memory fallback
        if room_id in self.in_memory_rooms and self.in_memory_rooms[room_id]['active']:
            return self.in_memory_rooms[room_id]
        return None

# Create global database instance
db = Database() 
