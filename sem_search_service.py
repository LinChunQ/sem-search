import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

from milvus_connection import get_collection, load_milvus_config


MODEL_NAME = "BAAI/bge-m3"
VECTOR_DIM = 1024
SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"ef": 64}}


class SemSearchService:
    def __init__(self):
        config = load_milvus_config()
        self.model_source = self._resolve_model_source(config)
        self._configure_huggingface_access(config)

        print(f"Loading embedding model from: {self.model_source}")
        self.model = SentenceTransformer(self.model_source)

        self.collection_name = str(
            config.get("MILVUS_COLLECTION_NAME", "claim_deduplication") or "claim_deduplication"
        ).strip()
        self.collection = get_collection(collection_name=self.collection_name, dim=VECTOR_DIM)

    @property
    def model_name(self) -> str:
        return self.model_source

    def _configure_huggingface_access(self, config: dict) -> None:
        hf_endpoint = str(config.get("HF_ENDPOINT", "https://hf-mirror.com") or "").strip()
        hf_token = str(config.get("HF_TOKEN", "") or "").strip()
        disable_symlink_warning = str(
            config.get("HF_HUB_DISABLE_SYMLINKS_WARNING", "1") or "1"
        ).strip()

        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        if disable_symlink_warning:
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = disable_symlink_warning

    def _resolve_model_source(self, config: dict) -> str:
        local_model_dir = str(config.get("EMBEDDING_MODEL_PATH", "") or "").strip()
        remote_model_name = str(config.get("EMBEDDING_MODEL_NAME", MODEL_NAME) or MODEL_NAME).strip()

        if local_model_dir:
            local_path = Path(local_model_dir)
            if local_path.exists():
                return str(local_path)

        return remote_model_name

    def add_contents(self, texts: list[str]) -> int:
        embeddings = self.model.encode(texts)
        entities = [texts, embeddings.tolist()]
        insert_result = self.collection.insert(entities)
        self.collection.flush()
        return insert_result.insert_count

    def search_contents(self, query: str, top_k: int) -> list[dict]:
        query_vector = self.model.encode([query]).tolist()
        self.collection.load()
        results = self.collection.search(
            data=query_vector,
            anns_field="vector",
            param=SEARCH_PARAMS,
            limit=top_k,
            output_fields=["content"],
        )

        hits = []
        for hit in results[0]:
            entity = hit.entity
            hits.append(
                {
                    "id": hit.id,
                    "score": float(hit.score),
                    "content": entity.get("content") if entity else "",
                }
            )
        return hits

    def build_page_context(self) -> dict:
        return {
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "add_message": None,
            "search_message": None,
            "search_results": [],
            "search_query": "",
            "top_k": 5,
            "add_text": "",
        }

    def health_payload(self) -> dict:
        return {
            "status": "ok",
            "collection_name": self.collection_name,
            "model_name": self.model_name,
        }
