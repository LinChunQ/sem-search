from pathlib import Path

from dotenv import dotenv_values
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    db,
    utility,
)


DEFAULT_ALIAS = "default"
CONFIG_FILE = Path(__file__).with_name(".env")
DEFAULT_INDEX_PARAMS = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 64},
}


def _config_flag(config: dict, name: str, default: bool = False) -> bool:
    value = config.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_milvus_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Milvus config file not found: {CONFIG_FILE}. "
            "Create .env and fill in your remote Milvus settings."
        )

    raw_config = dotenv_values(CONFIG_FILE)
    return {key: value for key, value in raw_config.items() if key is not None}


def connect_to_milvus(alias: str = DEFAULT_ALIAS) -> str:
    config = load_milvus_config()

    uri = str(config.get("MILVUS_URI", "") or "").strip()
    host = str(config.get("MILVUS_HOST", "localhost") or "localhost").strip()
    port = str(config.get("MILVUS_PORT", "19530") or "19530").strip()
    token = str(config.get("MILVUS_TOKEN", "") or "").strip()
    user = str(config.get("MILVUS_USER", "") or "").strip()
    password = str(config.get("MILVUS_PASSWORD", "") or "").strip()
    db_name = str(config.get("MILVUS_DB_NAME", "default") or "default").strip() or "default"

    connect_kwargs = {}
    if uri:
        connect_kwargs["uri"] = uri
    else:
        connect_kwargs["host"] = host
        connect_kwargs["port"] = port

    if token:
        connect_kwargs["token"] = token
    else:
        if user:
            connect_kwargs["user"] = user
        if password:
            connect_kwargs["password"] = password

    if _config_flag(config, "MILVUS_SECURE"):
        connect_kwargs["secure"] = True

    connections.connect(alias=alias, db_name="default", **connect_kwargs)

    if db_name != "default":
        existing_databases = db.list_database(using=alias)
        if db_name not in existing_databases:
            db.create_database(db_name, using=alias)
        db.using_database(db_name, using=alias)

    return alias


def get_collection(
    collection_name: str,
    dim: int,
    alias: str = DEFAULT_ALIAS,
    drop_existing: bool = False,
) -> Collection:
    connect_to_milvus(alias=alias)

    if drop_existing and utility.has_collection(collection_name, using=alias):
        utility.drop_collection(collection_name, using=alias)

    if utility.has_collection(collection_name, using=alias):
        return Collection(name=collection_name, using=alias)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Complaint deduplication collection")

    collection = Collection(name=collection_name, schema=schema, using=alias)
    collection.create_index(field_name="vector", index_params=DEFAULT_INDEX_PARAMS)
    return collection
