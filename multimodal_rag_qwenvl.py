#!/usr/bin/env python3
"""多模态 RAG 示例程序（Qwen-VL 后端）。

功能：
1. 构建本地知识库索引（文本 + 图片说明）。
2. 通过向量检索召回相关内容。
3. 将召回上下文与图片一并发给 Qwen-VL 生成答案。

默认使用 OpenAI 兼容接口（可接入 DashScope/vLLM/自建网关）。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class Chunk:
    kind: str  # "text" | "image"
    source: str
    content: str  # 文本块 / 图片说明


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (TEXT_EXTENSIONS | IMAGE_EXTENSIONS):
            yield path


def read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def to_data_url(image_path: Path) -> str:
    mime = "image/jpeg"
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".bmp":
        mime = "image/bmp"

    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


class QwenVLClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def caption_image(self, image_path: Path) -> str:
        prompt = "请简洁描述这张图片的主体、场景与关键信息，用于后续检索。"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": to_data_url(image_path)}},
                ],
            }
        ]
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.2)
        return (resp.choices[0].message.content or "").strip()

    def answer(self, query: str, context_text: str, image_paths: List[Path]) -> str:
        content = [
            {
                "type": "text",
                "text": (
                    "你是一个多模态RAG助手。请严格基于给定上下文回答。"
                    "如果上下文不足，请明确说不知道并指出缺失信息。\n\n"
                    f"【检索到的文本上下文】\n{context_text}\n\n"
                    f"【用户问题】\n{query}"
                ),
            }
        ]

        for p in image_paths:
            content.append({"type": "image_url", "image_url": {"url": to_data_url(p)}})

        messages = [{"role": "user", "content": content}]
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.3)
        return (resp.choices[0].message.content or "").strip()


def build_chunks(knowledge_dir: Path, qwen: QwenVLClient | None) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in iter_files(knowledge_dir):
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            text = read_text_file(path)
            for c in chunk_text(text):
                chunks.append(Chunk(kind="text", source=str(path), content=c))
        elif suffix in IMAGE_EXTENSIONS:
            if qwen is None:
                continue
            caption = qwen.caption_image(path)
            if caption:
                chunks.append(Chunk(kind="image", source=str(path), content=caption))
    return chunks


def build_index(knowledge_dir: Path, index_dir: Path, embed_model: str, qwen: QwenVLClient | None) -> None:
    chunks = build_chunks(knowledge_dir, qwen)
    if not chunks:
        raise RuntimeError("未生成任何可索引内容，请检查知识库目录。")

    embedder = SentenceTransformer(embed_model)
    embeddings = embedder.encode([c.content for c in chunks], normalize_embeddings=True)
    vec = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(vec.shape[1])
    index.add(vec)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "vectors.faiss"))

    metadata = {
        "embed_model": embed_model,
        "dim": int(vec.shape[1]),
        "chunks": [asdict(c) for c in chunks],
    }
    (index_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(index_dir: Path) -> Tuple[faiss.Index, dict]:
    index = faiss.read_index(str(index_dir / "vectors.faiss"))
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    return index, metadata


def retrieve(query: str, index: faiss.Index, metadata: dict, top_k: int, embed_model: str) -> List[dict]:
    embedder = SentenceTransformer(embed_model)
    q = embedder.encode([query], normalize_embeddings=True)
    qvec = np.asarray(q, dtype=np.float32)
    distances, idxs = index.search(qvec, top_k)

    result: List[dict] = []
    for score, idx in zip(distances[0], idxs[0]):
        if idx < 0:
            continue
        item = metadata["chunks"][idx]
        item["score"] = float(score)
        result.append(item)
    return result


def cmd_index(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    qwen = None
    if api_key and args.enable_image_caption:
        qwen = QwenVLClient(base_url=args.base_url, api_key=api_key, model=args.model)

    build_index(
        knowledge_dir=Path(args.knowledge_dir),
        index_dir=Path(args.index_dir),
        embed_model=args.embed_model,
        qwen=qwen,
    )
    print(f"索引完成，输出目录: {args.index_dir}")


def cmd_ask(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("请通过 --api-key 或 OPENAI_API_KEY 提供 Qwen-VL 服务 API Key。")

    qwen = QwenVLClient(base_url=args.base_url, api_key=api_key, model=args.model)
    index, metadata = load_index(Path(args.index_dir))

    hits = retrieve(args.query, index, metadata, top_k=args.top_k, embed_model=metadata["embed_model"])
    context_text = "\n\n".join(
        f"[score={h['score']:.3f}] ({h['kind']}) {h['source']}\n{h['content']}" for h in hits if h["kind"] == "text"
    )

    image_paths: List[Path] = []
    for h in hits:
        if h["kind"] == "image":
            p = Path(h["source"])
            if p.exists():
                image_paths.append(p)
    image_paths = image_paths[: args.max_images]

    answer = qwen.answer(args.query, context_text=context_text, image_paths=image_paths)

    print("\n=== 检索结果 ===")
    for h in hits:
        print(f"- {h['kind']} | {h['score']:.3f} | {h['source']}")
    print("\n=== 回答 ===")
    print(answer)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多模态 RAG（Qwen-VL）")
    sub = p.add_subparsers(required=True)

    p_index = sub.add_parser("index", help="构建知识库索引")
    p_index.add_argument("--knowledge-dir", required=True, help="知识库目录（文本与图片）")
    p_index.add_argument("--index-dir", default="./index_store", help="索引输出目录")
    p_index.add_argument("--embed-model", default="BAAI/bge-m3", help="向量模型")
    p_index.add_argument("--enable-image-caption", action="store_true", help="启用图片自动描述（需要API）")
    p_index.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="OpenAI兼容接口地址")
    p_index.add_argument("--model", default="qwen-vl-plus", help="Qwen-VL 模型名")
    p_index.add_argument("--api-key", default="", help="API Key（可选，默认读取OPENAI_API_KEY）")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="检索并回答")
    p_ask.add_argument("--index-dir", default="./index_store", help="索引目录")
    p_ask.add_argument("--query", required=True, help="用户问题")
    p_ask.add_argument("--top-k", type=int, default=5, help="召回条数")
    p_ask.add_argument("--max-images", type=int, default=2, help="注入回答模型的图片数量")
    p_ask.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="OpenAI兼容接口地址")
    p_ask.add_argument("--model", default="qwen-vl-plus", help="Qwen-VL 模型名")
    p_ask.add_argument("--api-key", default="", help="API Key（可选，默认读取OPENAI_API_KEY）")
    p_ask.set_defaults(func=cmd_ask)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
