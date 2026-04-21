import os
from pathlib import Path
from typing import Any

from pymilvus import AnnSearchRequest, WeightedRanker
from sentence_transformers import SentenceTransformer

from milvus_connection import get_collection, load_milvus_config


try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    BGEM3FlagModel = None


MODEL_NAME = "BAAI/bge-m3"
VECTOR_DIM = 1024
DENSE_SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"ef": 64}}
SPARSE_SEARCH_PARAMS = {"metric_type": "IP", "params": {"drop_ratio_search": 0.1}}


def _config_flag(config: dict, name: str, default: bool = False) -> bool:
    value = config.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _config_float(config: dict, name: str, default: float) -> float:
    value = config.get(name)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _config_int(config: dict, name: str, default: int) -> int:
    value = config.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


class SemSearchService:
    def __init__(self):
        config = load_milvus_config()
        self.search_mode = str(config.get("SEARCH_MODE", "dense") or "dense").strip().lower()
        self.hybrid_enabled = self.search_mode == "hybrid"

        self.model_source = self._resolve_model_source(config)
        self.using_local_model = Path(self.model_source).exists()
        self._configure_model_cache(config)
        self._configure_huggingface_access(config)

        self.use_fp16 = _config_flag(config, "BGE_M3_USE_FP16", False)
        self.dense_weight = _config_float(config, "HYBRID_DENSE_WEIGHT", 0.7)
        self.sparse_weight = _config_float(config, "HYBRID_SPARSE_WEIGHT", 0.3)
        self.hybrid_candidate_limit = _config_int(config, "HYBRID_CANDIDATE_LIMIT", 20)

        print(f"Loading embedding model from: {self.model_source}")
        if self.hybrid_enabled:
            if BGEM3FlagModel is None:
                raise ImportError(
                    "SEARCH_MODE=hybrid requires the FlagEmbedding package. "
                    "Install it in the virtual environment first."
                )
            self.model = BGEM3FlagModel(self.model_source, use_fp16=self.use_fp16)
        else:
            self.model = SentenceTransformer(
                self.model_source,
                local_files_only=self.using_local_model,
            )

        self.collection_name = str(
            config.get("MILVUS_COLLECTION_NAME", "claim_deduplication") or "claim_deduplication"
        ).strip()
        self.recreate_collection = _config_flag(config, "MILVUS_RECREATE_COLLECTION", False)
        self.collection = get_collection(
            collection_name=self.collection_name,
            dim=VECTOR_DIM,
            drop_existing=self.recreate_collection,
            enable_hybrid=self.hybrid_enabled,
        )

    @property
    def model_name(self) -> str:
        return self.model_source

    def _configure_model_cache(self, config: dict) -> None:
        cache_root = str(config.get("HF_HOME", "") or "").strip()
        if cache_root:
            cache_path = Path(cache_root)
        else:
            cache_path = Path(__file__).resolve().parent / ".cache" / "huggingface"

        hub_cache = cache_path / "hub"
        transformers_cache = cache_path / "transformers"
        hub_cache.mkdir(parents=True, exist_ok=True)
        transformers_cache.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("HF_HOME", str(cache_path))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))

    def _configure_huggingface_access(self, config: dict) -> None:
        if self.using_local_model:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            return

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

        default_local_path = Path(__file__).resolve().parent / "models" / remote_model_name.split("/")[-1]
        if default_local_path.exists():
            return str(default_local_path)

        return remote_model_name

    def _encode_dense(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]

    def _encode_hybrid(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = [embedding.tolist() for embedding in output["dense_vecs"]]
        sparse_vectors = [self._normalize_sparse_vector(item) for item in output["lexical_weights"]]
        return dense_vectors, sparse_vectors

    def _normalize_sparse_vector(self, sparse_vector: dict[Any, Any]) -> dict[int, float]:
        normalized = {
            int(token_id): float(weight)
            for token_id, weight in sparse_vector.items()
            if float(weight) > 0.0
        }
        if not normalized:
            raise ValueError("BGE-M3 returned an empty sparse vector, which Milvus cannot index.")
        return normalized

    def add_contents(self, texts: list[str]) -> int:
        if self.hybrid_enabled:
            dense_vectors, sparse_vectors = self._encode_hybrid(texts)
            entities = [
                {
                    "content": text,
                    "vector": dense_vector,
                    "sparse_vector": sparse_vector,
                }
                for text, dense_vector, sparse_vector in zip(texts, dense_vectors, sparse_vectors)
            ]
        else:
            dense_vectors = self._encode_dense(texts)
            entities = [
                {"content": text, "vector": dense_vector}
                for text, dense_vector in zip(texts, dense_vectors)
            ]

        insert_result = self.collection.insert(entities)
        self.collection.flush()
        return insert_result.insert_count

    def search_contents(self, query: str, top_k: int) -> list[dict]:
        self.collection.load()

        if self.hybrid_enabled:
            dense_vectors, sparse_vectors = self._encode_hybrid([query])
            candidate_limit = max(top_k, self.hybrid_candidate_limit)

            dense_request = AnnSearchRequest(
                data=[dense_vectors[0]],
                anns_field="vector",
                param=DENSE_SEARCH_PARAMS,
                limit=candidate_limit,
            )
            sparse_request = AnnSearchRequest(
                data=[sparse_vectors[0]],
                anns_field="sparse_vector",
                param=SPARSE_SEARCH_PARAMS,
                limit=candidate_limit,
            )
            results = self.collection.hybrid_search(
                reqs=[dense_request, sparse_request],
                rerank=WeightedRanker(self.dense_weight, self.sparse_weight),
                limit=top_k,
                output_fields=["content"],
            )
        else:
            query_vector = self._encode_dense([query])
            results = self.collection.search(
                data=query_vector,
                anns_field="vector",
                param=DENSE_SEARCH_PARAMS,
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
            "search_mode": self.search_mode,
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
            "search_mode": self.search_mode,
            "hybrid_enabled": self.hybrid_enabled,
        }
