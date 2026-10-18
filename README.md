# AML Memory API

Agent Memory Challenge (Cycle 2) — Textual 文本记忆赛道 · 开源方法组。

实现 AML 统一 **Add / Search** 契约的长期记忆系统：负责记忆写入与检索，平台统一执行 Answer 与 Eval。

## 架构

```
平台 Add 请求 ─▶ POST /add ─▶ text-embedding-v4 向量化 ─▶ 存入 PostgreSQL(pgvector)
平台 Search 请求 ─▶ POST /search ─▶ 向量化(query) ─▶ 同 user_id 近邻检索 ─▶ top_k 证据 ─▶ 返回平台
```

## 接口

| 接口 | 方法 | 路径 | 鉴权 |
|---|---|---|---|
| 健康检查 | GET | `/health` | 无 |
| 写入记忆 | POST | `/add` | Bearer |
| 检索记忆 | POST | `/search` | Bearer |

字段严格对齐 AML 官方 API 合约，详见 `docs/接口设计文档.md`。

## 技术栈

- Python 3.10 + FastAPI + Uvicorn
- PostgreSQL 16 + pgvector（HNSW 余弦索引）
- embedding：阿里云 `text-embedding-v4`（1024 维，DashScope 协议）
- 特性：Bearer 鉴权、gzip 压缩、连接池、embedding 本地缓存

## 快速开始

```bash
# 1. 建库建表（见 docs/接口设计文档.md 第 8 节数据流 / 部署说明.md）
# 2. 安装依赖
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
vim .env  # 填 DASHSCOPE_API_KEY、API_KEY

# 4. 启动
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 目录

```
src/
├─ main.py           # 主程序
├─ requirements.txt  # 依赖
├─ .env.example      # 配置模板（.env 不提交）
└─ 部署说明.md        # 部署步骤
```
