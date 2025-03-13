from datetime import datetime
from typing import Dict, List, Optional
import json
import os

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    # Mock Pinecone module for testing
    from types import ModuleType
    from unittest.mock import MagicMock

    class ServerlessSpec:
        def __init__(self, cloud: str, region: str):
            self.cloud = cloud
            self.region = region

    class MockPinecone:
        def __init__(self, api_key: str, host: str = None):
            self.api_key = api_key
            self.host = host
            self.indexes = {}

        def create_index(
            self,
            name: str,
            dimension: int,
            spec: Optional[ServerlessSpec] = None,
            metric: str = "cosine",
        ):
            if name not in self.indexes:
                self.indexes[name] = MagicMock()
                self.indexes[name].upsert = MagicMock()
                self.indexes[name].query = MagicMock(return_value=MagicMock(matches=[]))

        def list_indexes(self):
            return list(self.indexes.keys())

        def delete_index(self, name: str):
            if name in self.indexes:
                del self.indexes[name]

        def Index(self, name: str):
            if name not in self.indexes:
                self.create_index(name, dimension=384)
            return self.indexes[name]

    Pinecone = MockPinecone


class UserBehaviorAnalyzer:
    def __init__(self, api_key: str, environment: str):
        """Initialize Pinecone for user behavior analysis"""
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "user-behavior"

        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Using text-embedding-ada-002 dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=environment),
            )

        self.index = self.pc.Index(self.index_name)
        self.storage_dir = "user_behavior_data"

        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    async def track_user_action(
        self,
        user_id: str,
        action_type: str,
        content_id: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Track a user action and store it in Pinecone"""
        timestamp = datetime.now().isoformat()

        # Create vector for the action
        action_data = {
            "user_id": user_id,
            "action_type": action_type,
            "content_id": content_id,
            "timestamp": timestamp,
        }

        if metadata:
            action_data.update(metadata)

        # Generate a unique ID for the action
        action_id = f"{user_id}_{action_type}_{timestamp}"

        # Store in Pinecone
        self.index.upsert(
            vectors=[
                (action_id, [0] * 384, action_data)
            ]  # Using placeholder vector for now
        )

        # Save to local storage
        self._save_to_storage(action_id, action_data)

        return action_id

    def get_user_actions(
        self,
        user_id: str,
        action_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict]:
        """Get user actions with optional filters"""
        filter_query = {"user_id": {"$eq": user_id}}

        if action_type:
            filter_query["action_type"] = {"$eq": action_type}

        if start_time:
            filter_query["timestamp"] = {"$gte": start_time.isoformat()}

        if end_time:
            if "timestamp" not in filter_query:
                filter_query["timestamp"] = {}
            filter_query["timestamp"]["$lte"] = end_time.isoformat()

        # Query Pinecone
        results = self.index.query(
            vector=[0] * 384,  # Using placeholder vector for now
            filter=filter_query,
            top_k=100,
        )

        return [match.metadata for match in results.matches]

    def analyze_user_behavior(self, user_id: str) -> Dict:
        """Analyze user behavior patterns"""
        actions = self.get_user_actions(user_id)

        # Count action types
        action_types = {}
        content_interactions = {}
        time_patterns = {"morning": 0, "afternoon": 0, "evening": 0, "night": 0}

        for action in actions:
            # Count action types
            action_type = action["action_type"]
            action_types[action_type] = action_types.get(action_type, 0) + 1

            # Count content interactions
            content_id = action["content_id"]
            content_interactions[content_id] = (
                content_interactions.get(content_id, 0) + 1
            )

            # Analyze time patterns
            timestamp = datetime.fromisoformat(action["timestamp"])
            hour = timestamp.hour

            if 5 <= hour < 12:
                time_patterns["morning"] += 1
            elif 12 <= hour < 17:
                time_patterns["afternoon"] += 1
            elif 17 <= hour < 22:
                time_patterns["evening"] += 1
            else:
                time_patterns["night"] += 1

        return {
            "total_actions": len(actions),
            "action_types": action_types,
            "content_interactions": content_interactions,
            "time_patterns": time_patterns,
        }

    def _save_to_storage(self, action_id: str, action_data: Dict):
        """Save action data to local storage"""
        file_path = os.path.join(self.storage_dir, f"{action_id}.json")
        with open(file_path, "w") as f:
            json.dump(action_data, f, indent=2)

    def cleanup(self):
        """Clean up Pinecone resources"""
        if self.index_name in self.pc.list_indexes():
            self.pc.delete_index(self.index_name)
