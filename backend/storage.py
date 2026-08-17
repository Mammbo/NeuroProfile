#Daniel Alvarez 
#8/16/26

# storage.py 
#store data in a vector database for search and pulling for dashboard

import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION: str = "videos_v1"
VECTOR_SIZE = 360 # summary_vec dim

PAYLOAD_FIELDS = (
    "video_id", "title", "duration", "system_profile",
    "system_names", "system_tiers", "system_derived",
    "moments", "timeline_path", "transcript"
)

def _pid(video_id: str) -> str:
    """Deterministic UUID from a video id (Qdrant rejects arbitrary string ids)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, video_id))

def _clean_vector(vector):
    # not excatly a clean up but NaN are benigng dead dims that are unaffected so pin them to 0
    # infinities if we every do get one are genuinely broken vectors so pin those to 0 to not mess up the cosine norm.
    a = np.nan_to_num(np.asarray(vector, dtype=np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)
    return a.tolist()

def build_payload(*, video_id, title, duration, system_profile, system_names, system_tiers, system_derived, moments, timeline_path, transcript):
    # all fields required, makes sure the payload is correct
    return {
        "video_id": video_id,
        "title": title,
        "duration": duration,
        "system_profile": list(system_profile),
        "system_names": list(system_names),
        "system_tiers": list(system_tiers),
        "system_derived": list(system_derived),
        "moments": moments,
        "timeline_path": timeline_path,
        "transcript": transcript,
    }
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
            points=[PointStruct(id=_pid(video_id), vector=_clean_vector(vector), payload=payload)],
        )
    
    def get_video(self, video_id):
        # fetch one stored video's payload by id none if absent
        recs = self.client.retrieve(self.videos_v1, ids=[_pid(video_id)], with_payload=True)
        return recs[0].payload if recs else None
    
    def search_similar(self, vector, limit=5, exclude_id=None):
        #Nearest neighbours by cosine. exclude_id drops the query video itself.
        hits = self.client.query_points(
            self.videos_v1, query=list(vector), limit=limit + 1, with_payload=True
        ).points
        out = [
            {"video_id": h.payload.get("video_id"), "score": h.score, "payload": h.payload}
            for h in hits
            if h.payload.get("video_id") != exclude_id
        ]
        return out[:limit]
    
    def delete_video(self, video_id):
        self.client.delete(self.videos_v1, points_selector=[_pid(video_id)])

    def reset(self):
        """Drop and recreate the collection — the 'wipe and re-encode' button."""
        self.client.delete_collection(self.videos_v1)
        self._ensure_collection(VECTOR_SIZE)


