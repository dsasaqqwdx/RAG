"""
Tracks which files (by content hash) have already been ingested, to prevent
duplicate embedding/storage when the same PDF is uploaded more than once.
"""
import hashlib
import json
import os

MANIFEST_PATH = "data/ingested_manifest.json"


def hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def record_ingestion(file_hash: str, filename: str, chunk_count: int):
    manifest = load_manifest()
    manifest[file_hash] = {"filename": filename, "chunks": chunk_count}
    save_manifest(manifest)


def find_existing(file_hash: str):
    """Returns the existing manifest entry if this exact content was already ingested, else None."""
    manifest = load_manifest()
    return manifest.get(file_hash)


def list_ingested_documents():
    """Returns a list of {filename, chunks} for every distinct file ever ingested."""
    manifest = load_manifest()
    return [{"filename": v["filename"], "chunks": v["chunks"]} for v in manifest.values()]