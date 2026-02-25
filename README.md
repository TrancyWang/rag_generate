# rag_generate

一个基于 **Qwen-VL（qwenv-vl）** 的多模态 RAG 示例程序。

## 功能
- 文本知识入库：支持 `txt/md/pdf`。
- 图片知识入库：可调用 Qwen-VL 自动生成图片说明，并参与向量检索。
- 检索增强问答：将召回文本 + 相关图片一起传给 Qwen-VL 生成答案。

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

如需自定义网关地址，可传入 `--base-url`。

## 1）构建索引
> `--enable-image-caption` 会对图片调用 Qwen-VL 生成说明。若不加该参数，图片不会入库。

```bash
python multimodal_rag_qwenvl.py index \
  --knowledge-dir ./knowledge \
  --index-dir ./index_store \
  --enable-image-caption \
  --model qwen-vl-plus
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
- 向量模型默认使用 `BAAI/bge-m3`，可通过 `--embed-model` 替换。
- 若只做纯文本 RAG，可在建库时不加 `--enable-image-caption`。
- 该示例偏工程骨架，生产环境建议补充：重排、权限过滤、缓存、评估与监控。
