#Daniel Alvarez 
#8/16/26

# storage.py 
#store data in a vector database for search and pulling for dashboard

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION: str = "videos_v1"
VECTOR_SIZE = 360 # summary_vec dim
def _pid(video_id: str) -> str:
    """Deterministic UUID from a video id (Qdrant rejects arbitrary string ids)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, video_id))

# init client
class QdrantVecDB():
    def __init__(self, collection=COLLECTION, size=VECTOR_SIZE, path="./qdrant_data"):
        self.client = QdrantClient(path=path)
        self.videos_v1 = collection
        self._ensure_collection(size)

    def _ensure_collection(self, size):
        if not self.client.collection_exists(self.videos_v1):
            self.client.create_collection(
                collection_name=self.videos_v1,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE)
            )
    # make uploading process for all vectors a queue
    def upsert_video(self, video_id, vector, payload=None):
        payload = dict(payload or {})
        payload["video_id"] = video_id          # keep the real id searchable in payload
        return self.client.upsert(
            collection_name=self.videos_v1,
            wait=True,
            points=[PointStruct(id=_pid(video_id), vector=list(vector), payload=payload)],
        )
    
    # add get_video and search later
    


