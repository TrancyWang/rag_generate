# rag_generate

一个基于 **Qwen-VL（qwenv-vl）** 的多模态 RAG 示例程序，支持主流向量库后端：
- **FAISS**（本地轻量）
- **Chroma**（本地持久化）
- **Milvus**（服务化向量数据库）

## 功能
- 文本知识入库：支持 `txt/md/pdf`。
- 图片知识入库：可调用 Qwen-VL 自动生成图片说明，并参与向量检索。
- 检索增强问答：将召回文本 + 相关图片一起传给 Qwen-VL 生成答案。
- 向量库可切换：`--vector-db faiss|chroma|milvus`。

## 安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 准备 API
程序默认通过 OpenAI 兼容接口访问 Qwen-VL（默认示例为阿里云 DashScope）：

```bash
export OPENAI_API_KEY="你的key"
```

## 1）构建索引
> `--enable-image-caption` 会对图片调用 Qwen-VL 生成说明。若不加该参数，图片不会入库。

### FAISS
```bash
python multimodal_rag_qwenvl.py index \
  --knowledge-dir ./knowledge \
  --index-dir ./index_store \
  --vector-db faiss \
  --enable-image-caption \
  --model qwen-vl-plus
```

### Chroma
```bash
python multimodal_rag_qwenvl.py index \
  --knowledge-dir ./knowledge \
  --index-dir ./index_store_chroma \
  --vector-db chroma \
  --collection multimodal_rag \
  --enable-image-caption
```

### Milvus
```bash
python multimodal_rag_qwenvl.py index \
  --knowledge-dir ./knowledge \
  --index-dir ./index_store_milvus \
  --vector-db milvus \
  --collection multimodal_rag \
  --milvus-uri http://localhost:19530 \
  --enable-image-caption
```

## 2）提问
```bash
python multimodal_rag_qwenvl.py ask \
  --index-dir ./index_store \
  --query "图里提到的设备用途是什么？" \
  --top-k 5 \
  --max-images 2 \
  --model qwen-vl-plus
```

## 目录建议
```text
knowledge/
  manual.md
  report.pdf
  scene.jpg
```

## 说明
- 向量模型默认 `BAAI/bge-m3`，可通过 `--embed-model` 替换。
- 若只做纯文本 RAG，可在建库时不加 `--enable-image-caption`。
- `ask` 会自动读取 `index_dir/metadata.json`，识别建库时使用的向量后端。
