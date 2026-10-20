"""
Agent Memory Challenge (Cycle 2) — Add / Search API
Textual 文本记忆赛道 · Bearer 鉴权 · PostgreSQL + pgvector

接口：
  GET  /health  健康检查（无需鉴权）
  POST /add     写入记忆（Bearer 鉴权）
  POST /search  检索记忆（Bearer 鉴权）

字段严格对齐 AML 官方 API 合约。

v1.1 优化：
  - 存消息真实时间戳 ts，检索按时间辅助
  - 语义去重：同 user_id 下相似内容覆盖/跳过（memory governance）
  - Search 带出同 session 相邻消息，拼上下文块
  - embedding 批量调用
"""

import os
import time
import math
import hashlib
import logging
from typing import List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
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

# 语义去重阈值：新记忆与库内已有记忆余弦相似度 > 该值时，视为重复，跳过写入
DEDUP_SIM_THRESHOLD = float(os.getenv("DEDUP_SIM_THRESHOLD", "0.95"))
# 同 session 去重阈值更高：流式增量是同一话题的合法延续，只有近乎完全相同才去重，避免误伤
DEDUP_SAME_SESSION_THRESHOLD = float(os.getenv("DEDUP_SAME_SESSION_THRESHOLD", "0.98"))

# ---------- App ----------
app = FastAPI(title="AML Memory API", version="1.1.0")
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


def _fetch_embeddings(texts: List[str]) -> List[List[float]]:
    """批量调用阿里云 text-embedding-v4，返回向量列表。带本地缓存。"""
    results: List[List[float]] = []
    to_fetch: List[str] = []
    to_fetch_idx: List[int] = []

    for i, t in enumerate(texts):
        if t in _embed_cache:
            results.append(_embed_cache[t])
        else:
            results.append([])  # 占位
            to_fetch.append(t)
            to_fetch_idx.append(i)

    if not to_fetch:
        return results

    # 未配置真实 key 时，用单位向量占位（非全零，避免余弦相似度 NaN）
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY.startswith("change_me"):
        for i, t in zip(to_fetch_idx, to_fetch):
            vec = [0.0] * EMBEDDING_DIM
            vec[0] = 1.0
            results[i] = vec
            _embed_cache[t] = vec
        return results

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    # 每批 5 条，避免单请求文本过多触发 400（text-embedding-v4 批量有长度/条数限制）
    batch_size = 5
    for start in range(0, len(to_fetch), batch_size):
        batch_texts = to_fetch[start : start + batch_size]
        batch_idx = to_fetch_idx[start : start + batch_size]
        payload = {
            "model": EMBEDDING_MODEL,
            "input": {"texts": batch_texts},
            "parameters": {"dimension": EMBEDDING_DIM},
        }
        try:
            resp = requests.post(EMBEDDING_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            embs = data["output"]["embeddings"]
            for i, t, e in zip(batch_idx, batch_texts, embs):
                vec = e["embedding"]
                results[i] = vec
                _embed_cache[t] = vec
        except HTTPException:
            raise
        except Exception as e:
            logger.error("embedding 批量调用失败(批%d): %s", start, e)
            # 降级：逐条重试，保证单条失败不影响整体
            for i, t in zip(batch_idx, batch_texts):
                try:
                    single_payload = {
                        "model": EMBEDDING_MODEL,
                        "input": {"texts": [t]},
                        "parameters": {"dimension": EMBEDDING_DIM},
                    }
                    r2 = requests.post(EMBEDDING_URL, json=single_payload, headers=headers, timeout=30)
                    r2.raise_for_status()
                    vec = r2.json()["output"]["embeddings"][0]["embedding"]
                    results[i] = vec
                    _embed_cache[t] = vec
                except Exception as e2:
                    logger.error("单条 embedding 也失败: %s", e2)
                    raise HTTPException(status_code=502, detail={"reason": "embedding failed"})

    return results


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


# ---------- 混合检索工具 ----------
# 中文停用词（常见虚词，过滤以降低噪声）
_STOPWORDS = set(
    "的 了 是 在 我 你 他 她 它 我们 你们 他们 这 那 有 和 与 或 也 就 都 而 及 被 把 让 从 到 对 为 但 不 没 很 太 只 个 些 之 其 啊 呀 吗 呢 吧 什么 怎么 为什么 哪 谁 几 多 少 上下 前 后 里 外 中 年 月 日 今天 明天 昨天 现在 一个 一些 什么".split()
)


def _tokenize(text: str) -> set:
    """提取实词 token 集合：中文按字符 bigram，英文按空格分词，过滤停用词。"""
    tokens = set()
    # 英文/数字词（含中英混合，按非中文字符切）
    import re

    words = re.findall(r"[a-zA-Z0-9]+", text)
    for w in words:
        if len(w) > 1 and w.lower() not in _STOPWORDS:
            tokens.add(w.lower())

    # 中文：提取连续汉字串，做 bigram
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in chinese_runs:
        # 单字直接加入（除非是停用词）
        if len(run) == 1:
            if run not in _STOPWORDS:
                tokens.add(run)
        else:
            for i in range(len(run) - 1):
                bg = run[i : i + 2]
                if bg not in _STOPWORDS:
                    tokens.add(bg)
    return tokens


def _literal_overlap(query: str, content: str) -> float:
    """计算 query 与 content 的字面重叠度（Jaccard），∈[0,1]。"""
    q = _tokenize(query)
    c = _tokenize(content)
    if not q or not c:
        return 0.0
    inter = len(q & c)
    union = len(q | c)
    return inter / union if union else 0.0


# 混合检索权重
LAMBDA_LITERAL = float(os.getenv("LAMBDA_LITERAL", "0.15"))  # 字面重叠权重
MU_RECENCY = float(os.getenv("MU_RECENCY", "0.05"))  # 时间新鲜度权重
RECENCY_HALFLIFE_MS = float(os.getenv("RECENCY_HALFLIFE_MS", str(7 * 24 * 3600 * 1000)))  # 7 天半衰期


def _recency_bonus(ts: Optional[int], now_ms: int) -> float:
    """时间新鲜度加分，∈[0,1]，越新越高（指数衰减）。"""
    if ts is None:
        return 0.0
    age = max(0, now_ms - ts)
    return math.exp(-age / RECENCY_HALFLIFE_MS)


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

    # 批量 embedding（省请求次数）
    contents = [m.content for m in req.messages]
    vecs = _fetch_embeddings(contents)

    with pool.connection() as conn:
        register_vector(conn)
        for m, vec in zip(req.messages, vecs):
            # 时间戳：优先用消息自带 timestamp（Unix 毫秒），否则用当前时间
            ts = m.timestamp if m.timestamp is not None else int(time.time() * 1000)

            # 语义去重：查同 user_id 下最相似的一条（统一阈值，同 v1.1）
            dup = conn.execute(
                """
                SELECT id, 1 - (embedding <=> %s) AS sim
                FROM memories
                WHERE user_id = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s
                LIMIT 1
                """,
                (Vector(vec), req.user_id, Vector(vec)),
            ).fetchone()

            if dup is not None and dup[1] is not None and float(dup[1]) > DEDUP_SIM_THRESHOLD:
                # 视为重复：更新时间戳（覆盖旧记忆，体现"更新/遗忘"治理），不新增
                conn.execute(
                    "UPDATE memories SET ts = %s WHERE id = %s",
                    (ts, dup[0]),
                )
                continue

            conn.execute(
                """
                INSERT INTO memories (user_id, session_id, content, embedding, ts)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (req.user_id, req.session_id, m.content, Vector(vec), ts),
            )
        conn.commit()

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

    vec = _fetch_embeddings([req.query])[0]

    # 召回池扩大到 top_k * 3
    candidate_limit = req.top_k * 3

    with pool.connection() as conn:
        register_vector(conn)
        # 向量检索召回候选池
        rows = conn.execute(
            """
            SELECT id, content, ts, created_at, session_id,
                   1 - (embedding <=> %s) AS vec_score
            FROM memories
            WHERE user_id = %s AND embedding IS NOT NULL
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (Vector(vec), req.user_id, Vector(vec), candidate_limit),
        ).fetchall()

        # 扩召回：字面高度匹配的条目并入候选池（补充精确事实命中），
        # 但排序仍按向量分，不扰动 v1.1 已验证的排序逻辑
        seen_ids = [r[0] for r in rows]
        q_tokens = _tokenize(req.query)
        if q_tokens:
            extra = conn.execute(
                """
                SELECT id, content, ts, created_at, session_id,
                       1 - (embedding <=> %s) AS vec_score
                FROM memories
                WHERE user_id = %s AND embedding IS NOT NULL
                  AND NOT (id = ANY(%s))
                ORDER BY created_at DESC
                LIMIT 200
                """,
                (Vector(vec), req.user_id, seen_ids),
            ).fetchall()
            literal_hits = []
            for r in extra:
                content = r[1]
                c_tokens = _tokenize(content)
                if not c_tokens:
                    continue
                overlap = len(q_tokens & c_tokens) / len(q_tokens)
                if overlap >= 0.5:  # query 的实词一半以上出现在内容中 → 精确匹配候选
                    literal_hits.append(r)
            # 字面命中并入池子，最多补 top_k 条
            rows = list(rows) + literal_hits[: req.top_k]

        # 排序：纯向量分（v1.1 已验证），按分降序取 top_k
        rows = sorted(rows, key=lambda r: (float(r[5]) if r[5] is not None else 0.0), reverse=True)
        rows = rows[: req.top_k]

        # 带出同 session 上下文：按 ts 排序取「位置前后各 1 条」邻居（顺序语义正确）
        data = []
        for r in rows:
            mem_id, content, ts, created_at, session_id, vec_score = r
            if session_id and ts is not None:
                # 找该记忆在 session 内按 ts 排序的位置，取前后各 1 条
                seq = conn.execute(
                    """
                    SELECT id, content FROM memories
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY COALESCE(ts, 0) ASC, id ASC
                    """,
                    (req.user_id, session_id),
                ).fetchall()
                idx = None
                for i, (sid, _) in enumerate(seq):
                    if sid == mem_id:
                        idx = i
                        break
                if idx is not None:
                    neighbors = []
                    if idx > 0:
                        neighbors.append(seq[idx - 1][1])
                    if idx < len(seq) - 1:
                        neighbors.append(seq[idx + 1][1])
                    context = "\n".join(neighbors + [content])
                else:
                    context = content
            else:
                context = content

            s = float(vec_score) if vec_score is not None else None
            data.append(
                {
                    "id": str(mem_id),
                    "content": context,
                    "score": round(s, 6) if s is not None and math.isfinite(s) else None,
                    "created_at": created_at.isoformat() if created_at is not None else None,
                }
            )

    return JSONResponse({"data": data})
