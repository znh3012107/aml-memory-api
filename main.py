"""
Agent Memory Challenge (Cycle 2) — Add / Search API
Textual 文本记忆赛道 · Bearer 鉴权 · PostgreSQL + pgvector

接口：
  GET  /health  健康检查（无需鉴权）
  POST /add     写入记忆（Bearer 鉴权）
  POST /search  检索记忆（Bearer 鉴权）

字段严格对齐 AML 官方 API 合约。
"""

import os
import time
import logging
from typing import List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from pgvector.utils import Vector

load_dotenv()

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("aml")

# ---------- 配置 ----------
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "aml")
DB_USER = os.getenv("DB_USER", "aml")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

API_KEY = os.getenv("API_KEY", "change_me_to_platform_key")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_URL = os.getenv(
    "EMBEDDING_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
)

# ---------- App ----------
app = FastAPI(title="AML Memory API", version="1.0.0")
app.add_middleware(GZipMiddleware, minimum_size=256)  # 开 gzip 省流量


# ---------- 数据库连接池 ----------
_conninfo = (
    f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
    f"user={DB_USER} password={DB_PASSWORD}"
)
pool = ConnectionPool(_conninfo, min_size=1, max_size=10, open=False)


@app.on_event("startup")
def startup():
    pool.open()
    # 注册 vector 类型，确保 psycopg 能处理 vector 列
    with pool.connection() as conn:
        register_vector(conn)
    logger.info("数据库连接池已就绪")


@app.on_event("shutdown")
def shutdown():
    pool.close()


# ---------- 鉴权 ----------
def check_auth(authorization: Optional[str]) -> None:
    """Bearer 鉴权：校验 Authorization: Bearer <key>"""
    if not authorization:
        raise HTTPException(status_code=401, detail={"reason": "missing authorization"})
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail={"reason": "invalid auth scheme"})
    if parts[1] != API_KEY:
        raise HTTPException(status_code=401, detail={"reason": "invalid api key"})


# ---------- embedding ----------
_embed_cache: dict = {}  # 文本 -> 向量 的本地缓存，省 token + 省流量


def embed_text(text: str) -> List[float]:
    """调用阿里云 text-embedding-v4，返回向量。带本地缓存。"""
    if text in _embed_cache:
        return _embed_cache[text]

    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY.startswith("change_me"):
        # 未配置真实 key 时，返回单位向量占位（非全零，避免余弦相似度 NaN），便于接口结构联调
        vec = [0.0] * EMBEDDING_DIM
        vec[0] = 1.0
        _embed_cache[text] = vec
        return vec

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": {"texts": [text]},
        "parameters": {"dimension": EMBEDDING_DIM},
    }
    try:
        resp = requests.post(EMBEDDING_URL, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        vec = data["output"]["embeddings"][0]["embedding"]
        _embed_cache[text] = vec
        return vec
    except Exception as e:
        logger.error("embedding 调用失败: %s", e)
        raise HTTPException(status_code=502, detail={"reason": "embedding failed"})


# ---------- 请求模型 ----------
class Message(BaseModel):
    role: str
    content: str  # 文本赛道为字符串
    timestamp: Optional[int] = None


class AddRequest(BaseModel):
    request_id: str
    messages: List[Message]
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str
    options: Optional[List[str]] = None
    user_id: str
    top_k: int = Field(..., gt=0)


# ---------- 接口 ----------
@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@app.post("/add")
def add(req: AddRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)

    if not req.messages:
        raise HTTPException(status_code=422, detail={"reason": "messages is empty"})
    for m in req.messages:
        if not m.content or not m.content.strip():
            raise HTTPException(status_code=422, detail={"reason": "empty content"})

    with pool.connection() as conn:
        register_vector(conn)
        for m in req.messages:
            vec = embed_text(m.content)
            conn.execute(
                """
                INSERT INTO memories (user_id, session_id, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (req.user_id, req.session_id, m.content, Vector(vec)),
            )
        conn.commit()

    # 同步语义：写入完成（已 commit）后才返回
    return JSONResponse(
        {
            "success": True,
            "request_id": req.request_id,
            "user_id": req.user_id,
            "session_id": req.session_id,
        }
    )


@app.post("/search")
def search(req: SearchRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)

    vec = embed_text(req.query)

    with pool.connection() as conn:
        register_vector(conn)
        rows = conn.execute(
            """
            SELECT id, content, created_at, 1 - (embedding <=> %s) AS score
            FROM memories
            WHERE user_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (Vector(vec), req.user_id, Vector(vec), req.top_k),
        ).fetchall()

    import math

    data = []
    for r in rows:
        score = None
        if r[3] is not None:
            s = float(r[3])
            score = round(s, 6) if math.isfinite(s) else None
        data.append(
            {
                "id": str(r[0]),
                "content": r[1],
                "score": score,
                "created_at": r[2].isoformat() if r[2] is not None else None,
            }
        )

    return JSONResponse({"data": data})
