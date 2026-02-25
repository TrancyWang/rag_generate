#!/usr/bin/env python3
"""多模态 RAG 示例程序（Qwen-VL + 主流向量库）。"""

from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class Chunk:
    kind: str
    source: str
    content: str


def require_numpy():
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("请先安装 numpy：pip install numpy") from e
    return np


def require_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("请先安装 sentence-transformers：pip install sentence-transformers") from e
    return SentenceTransformer(model_name)


def require_openai_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("请先安装 openai：pip install openai") from e
    return OpenAI(base_url=base_url, api_key=api_key)


def require_pdf_reader(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("请先安装 pypdf：pip install pypdf") from e
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


class VectorStore:
    def add(self, vectors, chunks: List[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query_vec, top_k: int) -> List[dict]:
        raise NotImplementedError


class FaissStore(VectorStore):
    def __init__(self, index_dir: Path, metadata: dict):
        self.index_dir = index_dir
        self.metadata = metadata

    def add(self, vectors, chunks: List[Chunk]) -> None:
        try:
            import faiss
        except Exception as e:
            raise RuntimeError("请先安装 faiss-cpu：pip install faiss-cpu") from e

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_dir / "vectors.faiss"))
        self.metadata["chunks"] = [asdict(c) for c in chunks]
        (self.index_dir / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, query_vec, top_k: int) -> List[dict]:
        try:
            import faiss
        except Exception as e:
            raise RuntimeError("请先安装 faiss-cpu：pip install faiss-cpu") from e

        index = faiss.read_index(str(self.index_dir / "vectors.faiss"))
        meta = json.loads((self.index_dir / "metadata.json").read_text(encoding="utf-8"))
        distances, idxs = index.search(query_vec, top_k)
        hits = []
        for score, idx in zip(distances[0], idxs[0]):
            if idx < 0:
                continue
            item = dict(meta["chunks"][idx])
            item["score"] = float(score)
            hits.append(item)
        return hits


class ChromaStore(VectorStore):
    def __init__(self, index_dir: Path, metadata: dict):
        self.index_dir = index_dir
        self.metadata = metadata

    def _get_collection(self):
        try:
            import chromadb
        except Exception as e:
            raise RuntimeError("请先安装 chromadb：pip install chromadb") from e

        client = chromadb.PersistentClient(path=str(self.index_dir / "chroma_db"))
        name = self.metadata.get("collection", "multimodal_rag")
        return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def add(self, vectors, chunks: List[Chunk]) -> None:
        coll = self._get_collection()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        coll.upsert(
            ids=ids,
            embeddings=vectors.tolist(),
            documents=[c.content for c in chunks],
            metadatas=[{"kind": c.kind, "source": c.source} for c in chunks],
        )
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata["count"] = len(chunks)
        (self.index_dir / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, query_vec, top_k: int) -> List[dict]:
        coll = self._get_collection()
        ret = coll.query(query_embeddings=query_vec.tolist(), n_results=top_k, include=["documents", "metadatas", "distances"])
        hits: List[dict] = []
        for doc, meta, dist in zip(ret.get("documents", [[]])[0], ret.get("metadatas", [[]])[0], ret.get("distances", [[]])[0]):
            hits.append({"kind": meta.get("kind", "text"), "source": meta.get("source", ""), "content": doc, "score": 1.0 - float(dist)})
        return hits


class MilvusStore(VectorStore):
    def __init__(self, index_dir: Path, metadata: dict):
        self.index_dir = index_dir
        self.metadata = metadata

    def _connect(self):
        try:
            from pymilvus import connections
        except Exception as e:
            raise RuntimeError("请先安装 pymilvus：pip install pymilvus") from e
        connections.connect(alias="default", uri=self.metadata.get("milvus_uri", "http://localhost:19530"))

    def add(self, vectors, chunks: List[Chunk]) -> None:
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility
        except Exception as e:
            raise RuntimeError("请先安装 pymilvus：pip install pymilvus") from e
        self._connect()
        name = self.metadata.get("collection", "multimodal_rag")
        if utility.has_collection(name):
            utility.drop_collection(name)

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="kind", dtype=DataType.VARCHAR, max_length=16),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vectors.shape[1]),
            ]
        )
        coll = Collection(name=name, schema=schema)
        coll.insert([
            list(range(len(chunks))),
            [c.kind for c in chunks],
            [c.source[:1024] for c in chunks],
            [c.content[:8192] for c in chunks],
            vectors.tolist(),
        ])
        coll.create_index(field_name="embedding", index_params={"metric_type": "IP", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}})
        coll.load()

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata["count"] = len(chunks)
        (self.index_dir / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, query_vec, top_k: int) -> List[dict]:
        try:
            from pymilvus import Collection
        except Exception as e:
            raise RuntimeError("请先安装 pymilvus：pip install pymilvus") from e
        self._connect()
        coll = Collection(self.metadata.get("collection", "multimodal_rag"))
        coll.load()
        result = coll.search(
            data=query_vec.tolist(),
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["kind", "source", "content"],
        )
        hits: List[dict] = []
        for item in result[0]:
            ent = item.entity
            hits.append({"kind": ent.get("kind"), "source": ent.get("source"), "content": ent.get("content"), "score": float(item.distance)})
        return hits


def create_store(index_dir: Path, metadata: dict) -> VectorStore:
    backend = metadata.get("vector_db", "faiss")
    if backend == "faiss":
        return FaissStore(index_dir, metadata)
    if backend == "chroma":
        return ChromaStore(index_dir, metadata)
    if backend == "milvus":
        return MilvusStore(index_dir, metadata)
    raise ValueError(f"不支持的向量库: {backend}")


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (TEXT_EXTENSIONS | IMAGE_EXTENSIONS):
            yield path


def read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return require_pdf_reader(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return []
    out: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return out


def to_data_url(path: Path) -> str:
    mime = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower(), "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"


class QwenVLClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = require_openai_client(base_url, api_key)
        self.model = model

    def caption_image(self, image_path: Path) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[{"role": "user", "content": [{"type": "text", "text": "请简洁描述这张图片的主体、场景与关键信息，用于后续检索。"}, {"type": "image_url", "image_url": {"url": to_data_url(image_path)}}]}],
        )
        return (resp.choices[0].message.content or "").strip()

    def answer(self, query: str, context_text: str, image_paths: List[Path]) -> str:
        content = [{"type": "text", "text": f"你是一个多模态RAG助手。请严格基于上下文回答；若不足请说明不知道。\n\n【检索文本】\n{context_text}\n\n【用户问题】\n{query}"}]
        for p in image_paths:
            content.append({"type": "image_url", "image_url": {"url": to_data_url(p)}})
        resp = self.client.chat.completions.create(model=self.model, temperature=0.3, messages=[{"role": "user", "content": content}])
        return (resp.choices[0].message.content or "").strip()


def build_chunks(knowledge_dir: Path, qwen: QwenVLClient | None) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in iter_files(knowledge_dir):
        ext = path.suffix.lower()
        if ext in TEXT_EXTENSIONS:
            for c in chunk_text(read_text_file(path)):
                chunks.append(Chunk(kind="text", source=str(path), content=c))
        elif ext in IMAGE_EXTENSIONS and qwen is not None:
            cap = qwen.caption_image(path)
            if cap:
                chunks.append(Chunk(kind="image", source=str(path), content=cap))
    return chunks


def cmd_index(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    qwen = QwenVLClient(args.base_url, api_key, args.model) if (api_key and args.enable_image_caption) else None

    chunks = build_chunks(Path(args.knowledge_dir), qwen)
    if not chunks:
        raise RuntimeError("未生成任何索引内容，请检查知识目录。")

    np = require_numpy()
    embedder = require_embedder(args.embed_model)
    vectors = np.asarray(embedder.encode([c.content for c in chunks], normalize_embeddings=True), dtype=np.float32)

    metadata = {
        "vector_db": args.vector_db,
        "embed_model": args.embed_model,
        "dim": int(vectors.shape[1]),
        "collection": args.collection,
        "milvus_uri": args.milvus_uri,
    }
    create_store(Path(args.index_dir), metadata).add(vectors, chunks)
    print(f"索引完成: vector_db={args.vector_db}, index_dir={args.index_dir}")


def cmd_ask(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("请设置 --api-key 或 OPENAI_API_KEY。")

    index_dir = Path(args.index_dir)
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    np = require_numpy()
    embedder = require_embedder(metadata["embed_model"])
    qvec = np.asarray(embedder.encode([args.query], normalize_embeddings=True), dtype=np.float32)

    hits = create_store(index_dir, metadata).search(qvec, args.top_k)
    context_text = "\n\n".join(f"[score={h['score']:.3f}] ({h['kind']}) {h['source']}\n{h['content']}" for h in hits if h["kind"] == "text")
    image_paths = [Path(h["source"]) for h in hits if h["kind"] == "image" and Path(h["source"]).exists()][: args.max_images]

    answer = QwenVLClient(args.base_url, api_key, args.model).answer(args.query, context_text, image_paths)

    print("\n=== 检索结果 ===")
    for h in hits:
        print(f"- {h['kind']} | {h['score']:.3f} | {h['source']}")
    print("\n=== 回答 ===")
    print(answer)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多模态 RAG（Qwen-VL）")
    sub = p.add_subparsers(required=True)

    pi = sub.add_parser("index", help="构建索引")
    pi.add_argument("--knowledge-dir", required=True)
    pi.add_argument("--index-dir", default="./index_store")
    pi.add_argument("--vector-db", choices=["faiss", "chroma", "milvus"], default="faiss", help="向量库后端")
    pi.add_argument("--collection", default="multimodal_rag", help="chroma/milvus 集合名")
    pi.add_argument("--milvus-uri", default="http://localhost:19530", help="Milvus URI")
    pi.add_argument("--embed-model", default="BAAI/bge-m3")
    pi.add_argument("--enable-image-caption", action="store_true")
    pi.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    pi.add_argument("--model", default="qwen-vl-plus")
    pi.add_argument("--api-key", default="")
    pi.set_defaults(func=cmd_index)

    pa = sub.add_parser("ask", help="检索问答")
    pa.add_argument("--index-dir", default="./index_store")
    pa.add_argument("--query", required=True)
    pa.add_argument("--top-k", type=int, default=5)
    pa.add_argument("--max-images", type=int, default=2)
    pa.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    pa.add_argument("--model", default="qwen-vl-plus")
    pa.add_argument("--api-key", default="")
    pa.set_defaults(func=cmd_ask)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
