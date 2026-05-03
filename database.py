"""
数据库模块 —— 负责所有跟 PostgreSQL 打交道的事情
==============================================
包括：
- 创建表结构
- 存储对话记录
- 存储/检索记忆（带中文分词和加权排序）
"""

import os
import re
from typing import Optional, List

import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL", "")

HAS_PGVECTOR = False  # 在init_tables时检测

# Embedding 配置（向量搜索用）
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))

# 记忆向量搜索开关（需要同时设置 EMBEDDING_API_KEY）
MEMORY_VECTOR_ENABLED = os.getenv("MEMORY_VECTOR_ENABLED", "false").lower() == "true"

# 记忆搜索权重（纯关键词模式）
WEIGHT_KEYWORD = float(os.getenv("WEIGHT_KEYWORD", "0.5"))
WEIGHT_IMPORTANCE = float(os.getenv("WEIGHT_IMPORTANCE", "0.3"))
WEIGHT_RECENCY = float(os.getenv("WEIGHT_RECENCY", "0.2"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.15"))

# 记忆混合搜索权重（MEMORY_VECTOR_ENABLED=true 时生效）
MEMORY_HW_KEYWORD = float(os.getenv("MEMORY_HW_KEYWORD", "0.35"))
MEMORY_HW_SEMANTIC = float(os.getenv("MEMORY_HW_SEMANTIC", "0.35"))
MEMORY_HW_IMPORTANCE = float(os.getenv("MEMORY_HW_IMPORTANCE", "0.15"))
MEMORY_HW_RECENCY = float(os.getenv("MEMORY_HW_RECENCY", "0.15"))
MEMORY_SEMANTIC_THRESHOLD = float(os.getenv("MEMORY_SEMANTIC_THRESHOLD", "0.5"))


# ============================================================
# 连接池管理
# ============================================================

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL 未设置！")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5, statement_cache_size=0)
        print("✅ 数据库连接池已创建")
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("✅ 数据库连接池已关闭")


# ============================================================
# 表结构初始化
# ============================================================

async def init_tables():
    global HAS_PGVECTOR
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id              SERIAL PRIMARY KEY,
                session_id      TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT,
                model           TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                metadata        TEXT
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id              SERIAL PRIMARY KEY,
                content         TEXT NOT NULL,
                importance      INTEGER DEFAULT 5,
                source_session  TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                last_accessed   TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_fts 
            ON memories 
            USING gin(to_tsvector('simple', content));
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session 
            ON conversations (session_id, created_at);
        """)
        
        # 工具调用支持：加 metadata 字段（已有表自动迁移）
        await conn.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'conversations' AND column_name = 'metadata'
                ) THEN
                    ALTER TABLE conversations ADD COLUMN metadata TEXT;
                END IF;
            END $$;
        """)
        
        # content 允许 NULL（工具调用时 assistant 的 content 可能为空）
        await conn.execute("""
            ALTER TABLE conversations ALTER COLUMN content DROP NOT NULL;
        """)
        
        # 网关配置表（存储运行时可变配置）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS gateway_config (
                key     TEXT PRIMARY KEY,
                value   TEXT DEFAULT ''
            );
        """)
        
        # 分区缓存状态表（存储每个session的轮转状态）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS session_cache_state (
                session_id      TEXT PRIMARY KEY,
                summary         TEXT DEFAULT '',
                a_start_round   INTEGER DEFAULT 0,
                updated_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # 尝试启用pgvector扩展（向量搜索）
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            HAS_PGVECTOR = True
            print("✅ pgvector扩展已启用")
            
            # 对话表向量列
            await conn.execute(f"""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'conversations' AND column_name = 'embedding'
                    ) THEN
                        ALTER TABLE conversations ADD COLUMN embedding vector({EMBEDDING_DIM});
                    END IF;
                END $$;
            """)
            
            # 记忆表向量列
            await conn.execute(f"""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'memories' AND column_name = 'embedding'
                    ) THEN
                        ALTER TABLE memories ADD COLUMN embedding vector({EMBEDDING_DIM});
                    END IF;
                END $$;
            """)
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_embedding 
                    ON memories USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 10);
                """)
            except Exception:
                pass  # ivfflat需要一定行数才能建索引，初期跳过
        except Exception as e:
            HAS_PGVECTOR = False
            print(f"⚠️ pgvector不可用（{e}），向量搜索将使用Python端计算")
            
            # 回退：用TEXT列存JSON格式的向量
            await conn.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'conversations' AND column_name = 'embedding_json'
                    ) THEN
                        ALTER TABLE conversations ADD COLUMN embedding_json TEXT;
                    END IF;
                END $$;
            """)
            await conn.execute("""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'memories' AND column_name = 'embedding_json'
                    ) THEN
                        ALTER TABLE memories ADD COLUMN embedding_json TEXT;
                    END IF;
                END $$;
            """)
    
    print("✅ 数据库表结构已就绪")


# ============================================================
# 中文分词工具（基于 jieba）
# ============================================================

import jieba
import jieba.analyse

# 静默加载词典
jieba.setLogLevel(jieba.logging.INFO)

EN_WORD_PATTERN = re.compile(r'[a-zA-Z][a-zA-Z0-9]*')
NUM_PATTERN = re.compile(r'\d{2,}')
# 清理查询开头的时间戳（如 "2026-05-02 20:26"）
TIMESTAMP_PATTERN = re.compile(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*\d{1,2}:\d{1,2}\s*')

# 中文停用词（高频但无搜索价值的词）
_STOP_WORDS = frozenset({
    "的", "了", "在", "是", "我", "你", "他", "她", "它", "们",
    "这", "那", "有", "和", "与", "也", "都", "又", "就", "但",
    "而", "或", "到", "被", "把", "让", "从", "对", "为", "以",
    "及", "等", "个", "不", "没", "很", "太", "吗", "呢", "吧",
    "啊", "嗯", "哦", "哈", "呀", "嘛", "么", "啦", "哇", "喔",
    "会", "能", "要", "想", "去", "来", "说", "做", "看", "给",
    "上", "下", "里", "中", "大", "小", "多", "少", "好", "可以",
    "什么", "怎么", "如何", "哪里", "哪个", "为什么", "还是",
    "然后", "因为", "所以", "虽然", "但是", "可以", "已经",
    "一个", "一些", "一下", "一点", "一起", "一样",
    "比较", "应该", "可能", "如果", "这个", "那个",
    "自己", "知道", "觉得", "感觉", "时候", "现在",
})

# jieba 用户词典补充（默认词典缺失的词）
for _w in ["手账", "手帐", "搭子", "种草", "拔草", "安利", "内卷", "摆烂", "emo", "网关"]:
    jieba.add_word(_w)


def extract_search_keywords(query: str) -> List[str]:
    """
    从查询中提取搜索关键词（TF-IDF + 正则）

    1. 去掉开头的时间戳噪音
    2. 用 jieba.analyse.extract_tags (TF-IDF) 提取中文关键词
    3. 正则提取英文单词
    4. 保留4位以上数字（年份等，过滤短数字噪音）

    例如：
    "2026-05-02 20:26 写写手账看看书 放松大脑" → ["手账", "放松", "大脑"]
    "我昨天在手机上部署了Render然后吃了晚饭" → ["手机", "部署", "Render", "晚饭"]
    "春节干了什么" → ["春节"]
    "2026除夕"    → ["2026", "除夕"]
    """
    # 去掉时间戳前缀
    cleaned = TIMESTAMP_PATTERN.sub('', query).strip()
    if not cleaned:
        cleaned = query

    keywords = set()

    # 英文单词（2字符以上）
    for match in EN_WORD_PATTERN.finditer(cleaned):
        word = match.group()
        if len(word) >= 2:
            keywords.add(word)

    # 数字串（只保留4位以上，过滤 "05" "20" 这种时间噪音）
    for match in NUM_PATTERN.finditer(cleaned):
        num = match.group()
        if len(num) >= 4:
            keywords.add(num)

    # TF-IDF 关键词提取（比手动分词+停用词好很多）
    tags = jieba.analyse.extract_tags(cleaned, topK=10)
    for tag in tags:
        # 跳过纯英文/数字（已在上面处理）
        if EN_WORD_PATTERN.fullmatch(tag) or NUM_PATTERN.fullmatch(tag):
            continue
        if tag in _STOP_WORDS:
            continue
        keywords.add(tag)

    return list(keywords)


# ============================================================
# 向量搜索（OpenAI 兼容 Embedding API）
# ============================================================

async def compute_embedding(text: str) -> list:
    """调用 OpenAI 兼容的 Embedding API 计算文本向量"""
    if not EMBEDDING_API_KEY:
        return []
    
    try:
        import httpx
        
        if len(text) > 4000:
            text = text[:4000]
        
        body = {
            "model": EMBEDDING_MODEL,
            "input": text,
        }
        if EMBEDDING_DIM > 0:
            body["dimensions"] = EMBEDDING_DIM
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{EMBEDDING_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        print(f"⚠️ Embedding计算失败: {e}")
        return []


async def save_memory_embedding(conn, memory_id: int, embedding: list):
    """保存记忆向量到memories表"""
    if not embedding:
        return
    
    if HAS_PGVECTOR:
        vec_str = '[' + ','.join(str(f) for f in embedding) + ']'
        await conn.execute(
            "UPDATE memories SET embedding = $1::vector WHERE id = $2",
            vec_str, memory_id
        )
    else:
        import json
        await conn.execute(
            "UPDATE memories SET embedding_json = $1 WHERE id = $2",
            json.dumps(embedding), memory_id
        )


def _cosine_sim(a, b):
    """余弦相似度（纯Python）"""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def _min_max_normalize(scores: dict) -> dict:
    """min-max归一化到0-1"""
    if not scores:
        return {}
    vals = list(scores.values())
    min_v, max_v = min(vals), max(vals)
    spread = max_v - min_v
    if spread == 0:
        return {k: 1.0 for k in scores}
    return {k: (v - min_v) / spread for k, v in scores.items()}


# ============================================================
# 对话记录操作
# ============================================================

async def save_message(session_id: str, role: str, content: str, model: str = "", metadata: str = None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO conversations (session_id, role, content, model, metadata) VALUES ($1, $2, $3, $4, $5)",
            session_id, role, content, model, metadata,
        )


async def get_last_user_content(session_id: str) -> str:
    """获取指定session最后一条user消息的content"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT content FROM conversations
            WHERE session_id = $1 AND role = 'user'
            ORDER BY created_at DESC
            LIMIT 1
        """, session_id)
        return row['content'] if row else ""


async def update_last_assistant_message(session_id: str, new_content: str, model: str = ""):
    """覆盖指定session最后一条assistant消息的content（用于re-roll去重）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id FROM conversations
            WHERE session_id = $1 AND role = 'assistant'
            ORDER BY created_at DESC
            LIMIT 1
        """, session_id)
        if row:
            await conn.execute(
                "UPDATE conversations SET content = $1, model = $2 WHERE id = $3",
                new_content, model, row['id']
            )
            return True
        return False


async def get_recent_messages(session_id: str, limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content, metadata, created_at FROM conversations WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2",
            session_id, limit,
        )
        return list(reversed(rows))


async def search_conversations(query: str, limit: int = 20, offset: int = 0):
    """搜索对话内容，返回匹配的session列表"""
    keywords = extract_search_keywords(query)
    if not keywords:
        return [], 0
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        where_parts = []
        params = []
        for i, kw in enumerate(keywords):
            where_parts.append(f"c.content ILIKE '%' || ${i+1} || '%'")
            params.append(kw)
        where_clause = " OR ".join(where_parts)
        
        count_sql = f"""
            SELECT COUNT(DISTINCT c.session_id) as total
            FROM conversations c
            WHERE {where_clause}
        """
        total_row = await conn.fetchrow(count_sql, *params)
        total = total_row['total'] if total_row else 0
        
        if total == 0:
            return [], 0
        
        limit_idx = len(params) + 1
        offset_idx = len(params) + 2
        params.extend([limit, offset])
        
        sql = f"""
            WITH matched_sessions AS (
                SELECT DISTINCT c.session_id
                FROM conversations c
                WHERE {where_clause}
            ),
            session_info AS (
                SELECT 
                    ms.session_id,
                    MIN(c.created_at) as first_time,
                    MAX(c.created_at) as last_time,
                    COUNT(*) as message_count
                FROM matched_sessions ms
                JOIN conversations c ON c.session_id = ms.session_id
                GROUP BY ms.session_id
            )
            SELECT 
                si.session_id,
                si.first_time,
                si.last_time,
                si.message_count
            FROM session_info si
            ORDER BY si.last_time DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """
        rows = await conn.fetch(sql, *params)
        
        results = []
        for r in rows:
            results.append({
                'session_id': r['session_id'],
                'first_time': r['first_time'].isoformat() if r['first_time'] else None,
                'last_time': r['last_time'].isoformat() if r['last_time'] else None,
                'message_count': r['message_count'],
            })
        
        return results, total


async def update_message_content(message_id: int, new_content: str):
    """更新单条对话消息的内容"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE conversations SET content = $1 WHERE id = $2",
            new_content, message_id,
        )
        return int(result.split()[-1]) if result else 0


# ============================================================
# 记忆操作
# ============================================================

async def save_memory(content: str, importance: int = 5, source_session: str = ""):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO memories (content, importance, source_session) VALUES ($1, $2, $3) RETURNING id",
            content, importance, source_session,
        )
        
        # MEMORY_VECTOR_ENABLED 时自动计算 embedding
        if MEMORY_VECTOR_ENABLED and row:
            try:
                embedding = await compute_embedding(content)
                if embedding:
                    await save_memory_embedding(conn, row['id'], embedding)
            except Exception as e:
                print(f"⚠️ 记忆 {row['id']} embedding自动计算失败: {e}")


async def search_memories(query: str, limit: int = 10):
    """
    搜索相关记忆
    
    MEMORY_VECTOR_ENABLED=true 时走混合搜索（关键词 + 向量）
    否则走纯关键词搜索
    """
    if MEMORY_VECTOR_ENABLED:
        return await search_memories_hybrid(query, limit)
    
    # ---- 纯关键词搜索 ----
    keywords = extract_search_keywords(query)
    
    if not keywords:
        return []
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        # 每个关键词命中得1分
        case_parts = []
        params = []
        for i, kw in enumerate(keywords):
            case_parts.append(f"CASE WHEN content ILIKE '%' || ${i+1} || '%' THEN 1 ELSE 0 END")
            params.append(kw)
        
        hit_count_expr = " + ".join(case_parts)
        max_hits = len(keywords)
        
        # 至少命中一个关键词
        where_parts = [f"content ILIKE '%' || ${i+1} || '%'" for i in range(len(keywords))]
        where_clause = " OR ".join(where_parts)
        
        limit_idx = len(keywords) + 1
        params.append(limit)
        
        sql = f"""
            SELECT 
                id, content, importance, created_at,
                ({hit_count_expr}) AS hit_count,
                (
                    {WEIGHT_KEYWORD} * ({hit_count_expr})::float / {max_hits}.0 +
                    {WEIGHT_IMPORTANCE} * importance::float / 10.0 +
                    {WEIGHT_RECENCY} * (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0))
                ) AS score
            FROM memories
            WHERE {where_clause}
            ORDER BY score DESC, importance DESC, created_at DESC
            LIMIT ${limit_idx}
        """
        
        results = await conn.fetch(sql, *params)
        
        # 过滤低分记忆
        if MIN_SCORE_THRESHOLD > 0:
            before_count = len(results)
            results = [r for r in results if r['score'] >= MIN_SCORE_THRESHOLD]
            filtered = before_count - len(results)
        else:
            filtered = 0
        
        if results:
            print(f"🔍 搜索 '{query}' → 关键词 {keywords[:8]}{'...' if len(keywords)>8 else ''} → 命中 {len(results)} 条" + (f"（过滤 {filtered} 条低分）" if filtered else ""))
            for r in results[:3]:
                print(f"   📌 [score={r['score']:.3f}] (hits={r['hit_count']}, imp={r['importance']}) {r['content'][:60]}...")
            
            ids = [r["id"] for r in results]
            await conn.execute(
                "UPDATE memories SET last_accessed = NOW() WHERE id = ANY($1::int[])",
                ids,
            )
        else:
            print(f"🔍 搜索 '{query}' → 关键词 {keywords[:8]} → 无结果" + (f"（{filtered} 条被分数阈值过滤）" if filtered else ""))
        
        return results


async def search_memories_hybrid(query: str, limit: int = 10):
    """
    记忆混合搜索：关键词 + 向量，归一化后四维加权
    
    权重：MEMORY_HW_KEYWORD + MEMORY_HW_SEMANTIC + MEMORY_HW_IMPORTANCE + MEMORY_HW_RECENCY
    """
    from datetime import datetime, timezone
    
    keywords = extract_search_keywords(query)
    query_embedding = await compute_embedding(query) if EMBEDDING_API_KEY else []
    
    if not keywords and not query_embedding:
        return []
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        candidates = {}  # id -> {content, importance, created_at, kw_score, similarity}
        
        # ---- 关键词路 ----
        if keywords:
            case_parts = []
            params = []
            for i, kw in enumerate(keywords):
                case_parts.append(f"CASE WHEN content ILIKE '%' || ${i+1} || '%' THEN 1 ELSE 0 END")
                params.append(kw)
            
            hit_count_expr = " + ".join(case_parts)
            max_hits = len(keywords)
            where_parts = [f"content ILIKE '%' || ${i+1} || '%'" for i in range(len(keywords))]
            where_clause = " OR ".join(where_parts)
            
            limit_idx = len(keywords) + 1
            params.append(limit * 3)
            
            kw_sql = f"""
                SELECT id, content, importance, created_at,
                       ({hit_count_expr}) AS hit_count,
                       ({hit_count_expr})::float / {max_hits}.0 AS kw_score
                FROM memories
                WHERE {where_clause}
                ORDER BY kw_score DESC
                LIMIT ${limit_idx}
            """
            kw_rows = await conn.fetch(kw_sql, *params)
            
            for r in kw_rows:
                candidates[r['id']] = {
                    'content': r['content'],
                    'importance': r['importance'],
                    'created_at': r['created_at'],
                    'hit_count': r['hit_count'],
                    'kw_score': float(r['kw_score']),
                    'similarity': 0.0,
                }
        
        # ---- 向量路 ----
        if query_embedding:
            if HAS_PGVECTOR:
                vec_str = '[' + ','.join(str(f) for f in query_embedding) + ']'
                sem_rows = await conn.fetch("""
                    SELECT id, content, importance, created_at,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM memories
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                """, vec_str, limit * 3)
            else:
                # Python端计算cosine
                import json
                all_mem = await conn.fetch("""
                    SELECT id, content, importance, created_at, embedding_json
                    FROM memories WHERE embedding_json IS NOT NULL
                """)
                
                scored = []
                for row in all_mem:
                    try:
                        emb = json.loads(row['embedding_json'])
                        sim = _cosine_sim(query_embedding, emb)
                        scored.append({**dict(row), 'similarity': sim})
                    except Exception:
                        continue
                scored.sort(key=lambda x: -x['similarity'])
                sem_rows = scored[:limit * 3]
            
            for r in sem_rows:
                sim = float(r['similarity'])
                if sim < MEMORY_SEMANTIC_THRESHOLD:
                    continue
                mid = r['id']
                if mid in candidates:
                    candidates[mid]['similarity'] = sim
                else:
                    candidates[mid] = {
                        'content': r['content'],
                        'importance': r['importance'],
                        'created_at': r['created_at'],
                        'hit_count': 0,
                        'kw_score': 0.0,
                        'similarity': sim,
                    }
            
            # debug：向量路统计
            sem_total = len(sem_rows)
            sem_passed = sum(1 for r in sem_rows if float(r['similarity']) >= MEMORY_SEMANTIC_THRESHOLD)
            sem_max = max((float(r['similarity']) for r in sem_rows), default=0)
            if sem_total > 0 and sem_passed == 0:
                print(f"   🔢 向量路: {sem_total}条候选全被阈值过滤（最高sim={sem_max:.3f}, 阈值={MEMORY_SEMANTIC_THRESHOLD}）")
            elif sem_total > 0:
                print(f"   🔢 向量路: {sem_passed}/{sem_total}条通过阈值（最高sim={sem_max:.3f}）")
        
        if not candidates:
            print(f"🔍 混合搜索 '{query}' → 两路均无结果")
            return []
        
        # ---- 归一化 + 加权 ----
        kw_norm = _min_max_normalize({mid: v['kw_score'] for mid, v in candidates.items()})
        sem_norm = _min_max_normalize({mid: v['similarity'] for mid, v in candidates.items()})
        
        now = datetime.now(timezone.utc)
        final = []
        for mid, info in candidates.items():
            kw = kw_norm.get(mid, 0.0)
            sem = sem_norm.get(mid, 0.0)
            imp = info['importance'] / 10.0
            days = (now - info['created_at']).total_seconds() / 86400.0
            rec = 1.0 / (1.0 + days)
            
            score = (MEMORY_HW_KEYWORD * kw +
                     MEMORY_HW_SEMANTIC * sem +
                     MEMORY_HW_IMPORTANCE * imp +
                     MEMORY_HW_RECENCY * rec)
            
            final.append({
                'id': mid,
                'content': info['content'],
                'importance': info['importance'],
                'created_at': info['created_at'],
                'hit_count': info['hit_count'],
                'similarity': info['similarity'],
                'score': score,
            })
        
        final.sort(key=lambda x: (-x['score'], -x['importance']))
        
        # 过滤低分
        if MIN_SCORE_THRESHOLD > 0:
            before_count = len(final)
            final = [r for r in final if r['score'] >= MIN_SCORE_THRESHOLD]
            filtered = before_count - len(final)
        else:
            filtered = 0
        
        results = final[:limit]
        
        if results:
            mode_tag = "混合" if query_embedding else "关键词"
            kw_tag = f"关键词 {keywords[:6]}" if keywords else "无关键词"
            print(f"🔍 {mode_tag}搜索 '{query}' → {kw_tag} → 命中 {len(results)} 条" + (f"（过滤 {filtered} 条低分）" if filtered else ""))
            for r in results[:3]:
                print(f"   📌 [score={r['score']:.3f}] (kw={r['hit_count']}, sim={r['similarity']:.2f}, imp={r['importance']}) {r['content'][:60]}...")
            
            ids = [r["id"] for r in results]
            await conn.execute(
                "UPDATE memories SET last_accessed = NOW() WHERE id = ANY($1::int[])",
                ids,
            )
        else:
            print(f"🔍 混合搜索 '{query}' → 无结果" + (f"（{filtered} 条被过滤）" if filtered else ""))
        
        return [dict(r) for r in results]


async def get_pending_memory_embedding_count():
    """查询还没有embedding的记忆数量"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if HAS_PGVECTOR:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NULL AND content IS NOT NULL"
            )
        else:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM memories WHERE embedding_json IS NULL AND content IS NOT NULL"
            )


async def backfill_memory_embeddings(batch_size: int = 20):
    """给已有记忆补算embedding（没有embedding的记忆）"""
    if not EMBEDDING_API_KEY:
        print("⚠️ EMBEDDING_API_KEY 未设置，无法补算embedding")
        return 0
    
    pool = await get_pool()
    total_updated = 0
    
    async with pool.acquire() as conn:
        if HAS_PGVECTOR:
            rows = await conn.fetch("""
                SELECT id, content FROM memories 
                WHERE embedding IS NULL AND content IS NOT NULL
                ORDER BY id
                LIMIT $1
            """, batch_size)
        else:
            rows = await conn.fetch("""
                SELECT id, content FROM memories 
                WHERE embedding_json IS NULL AND content IS NOT NULL
                ORDER BY id
                LIMIT $1
            """, batch_size)
    
    if not rows:
        print("✅ 所有记忆已有embedding，无需补算")
        return 0
    
    print(f"🔄 开始补算记忆embedding... 本批 {len(rows)} 条")
    
    async with pool.acquire() as conn:
        for row in rows:
            try:
                embedding = await compute_embedding(row['content'] or '')
                if embedding:
                    await save_memory_embedding(conn, row['id'], embedding)
                    total_updated += 1
            except Exception as e:
                print(f"⚠️ 记忆 {row['id']} embedding计算失败: {e}")
    
    # 检查剩余
    async with pool.acquire() as conn:
        if HAS_PGVECTOR:
            remaining = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE embedding IS NULL AND content IS NOT NULL")
        else:
            remaining = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE embedding_json IS NULL AND content IS NOT NULL")
    
    print(f"✅ 本批补算完成：{total_updated}/{len(rows)} 条成功" + (f"，剩余 {remaining} 条待处理" if remaining > 0 else ""))
    return total_updated


async def get_recent_memories(limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT id, content, importance, created_at FROM memories ORDER BY created_at DESC LIMIT $1",
            limit,
        )


async def get_all_memories_count():
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM memories")
        return row["cnt"]


async def get_all_memories():
    """导出所有记忆（用于备份）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT content, importance, source_session, created_at FROM memories ORDER BY id"
        )
        return [dict(r) for r in rows]


async def get_all_memories_detail():
    """获取所有记忆（含 id，用于管理页面）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, content, importance, source_session, created_at FROM memories ORDER BY id"
        )
        return [dict(r) for r in rows]


async def update_memory(memory_id: int, content: str = None, importance: int = None):
    """更新单条记忆"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if content is not None and importance is not None:
            await conn.execute(
                "UPDATE memories SET content = $1, importance = $2 WHERE id = $3",
                content, importance, memory_id
            )
        elif content is not None:
            await conn.execute(
                "UPDATE memories SET content = $1 WHERE id = $2",
                content, memory_id
            )
        elif importance is not None:
            await conn.execute(
                "UPDATE memories SET importance = $1 WHERE id = $2",
                importance, memory_id
            )


async def delete_memory(memory_id: int):
    """删除单条记忆"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)


async def delete_memories_batch(memory_ids: list):
    """批量删除记忆"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM memories WHERE id = ANY($1::int[])", memory_ids
        )


# ============================================================
# 网关配置
# ============================================================

async def get_gateway_config(key: str, default: str = "") -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT value FROM gateway_config WHERE key = $1", key)
        return row['value'] if row else default


async def set_gateway_config(key: str, value: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO gateway_config (key, value) VALUES ($1, $2)
            ON CONFLICT (key) DO UPDATE SET value = $2
        """, key, value)


# ============================================================
# 对话历史读取（分区缓存用）
# ============================================================

async def get_conversation_messages(session_id: str, limit: int = 100):
    """按时间正序读取session的消息"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT role, content, metadata, created_at
            FROM conversations
            WHERE session_id = $1
            ORDER BY created_at ASC
            LIMIT $2
        """, session_id, limit)
        return [dict(r) for r in rows]


# ============================================================
# 分区缓存状态管理
# ============================================================

async def get_session_cache_state(session_id: str) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT summary, a_start_round, updated_at FROM session_cache_state WHERE session_id = $1",
            session_id
        )
        if row:
            raw_summary = row['summary'] or ''
            summary_parts = []
            if raw_summary:
                try:
                    import json
                    parsed = json.loads(raw_summary)
                    if isinstance(parsed, list):
                        summary_parts = parsed
                    else:
                        summary_parts = [raw_summary]
                except (json.JSONDecodeError, ValueError):
                    summary_parts = [raw_summary]
            return {
                'summary_parts': summary_parts,
                'a_start_round': row['a_start_round'] or 0,
                'updated_at': row['updated_at'],
            }
        return {'summary_parts': [], 'a_start_round': 0, 'updated_at': None}


async def save_session_cache_state(session_id: str, summary_parts: list, a_start_round: int):
    import json
    summary_json = json.dumps(summary_parts, ensure_ascii=False)
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO session_cache_state (session_id, summary, a_start_round, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (session_id) 
            DO UPDATE SET summary = $2, a_start_round = $3, updated_at = NOW()
        """, session_id, summary_json, a_start_round)


# ============================================================
# Token 使用记录
# ============================================================

async def ensure_token_usage_table():
    """确保token_usage表存在（在init_tables里调用）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id              SERIAL PRIMARY KEY,
                session_id      TEXT,
                model           TEXT,
                prompt_tokens   INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens    INTEGER DEFAULT 0,
                usage_type      TEXT DEFAULT 'chat',
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_usage_created ON token_usage (created_at DESC);
        """)


async def save_token_usage(session_id: str, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int, usage_type: str = "chat"):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO token_usage (session_id, model, prompt_tokens, completion_tokens, total_tokens, usage_type)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, session_id, model, prompt_tokens, completion_tokens, total_tokens, usage_type)


# ============================================================
# 对话记录管理
# ============================================================

async def ensure_conversation_titles_table():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_titles (
                session_id  TEXT PRIMARY KEY,
                title       TEXT DEFAULT ''
            );
        """)


async def get_conversations_paginated(page: int = 1, per_page: int = 20):
    offset = (page - 1) * per_page
    pool = await get_pool()
    async with pool.acquire() as conn:
        total_row = await conn.fetchrow(
            "SELECT COUNT(DISTINCT session_id) as total FROM conversations"
        )
        total = total_row['total'] if total_row else 0
        
        rows = await conn.fetch("""
            WITH session_info AS (
                SELECT session_id, MIN(created_at) as first_time, MAX(created_at) as last_time, COUNT(*) as message_count
                FROM conversations GROUP BY session_id ORDER BY last_time DESC LIMIT $1 OFFSET $2
            )
            SELECT si.*, ct.title as custom_title,
                   COALESCE(tu.total_all, 0) as total_tokens
            FROM session_info si
            LEFT JOIN conversation_titles ct ON si.session_id = ct.session_id
            LEFT JOIN (
                SELECT session_id, SUM(total_tokens) as total_all FROM token_usage WHERE usage_type = 'chat' GROUP BY session_id
            ) tu ON si.session_id = tu.session_id
            ORDER BY si.last_time DESC
        """, per_page, offset)
        
        results = []
        for r in rows:
            preview_row = await conn.fetchrow(
                "SELECT content FROM conversations WHERE session_id = $1 AND role = 'user' ORDER BY created_at LIMIT 1",
                r['session_id']
            )
            preview = preview_row['content'][:80] if preview_row else ''
            title = r['custom_title'] or (preview[:30] + '...' if len(preview) > 30 else preview) or r['session_id']
            results.append({
                'session_id': r['session_id'],
                'title': title,
                'first_time': r['first_time'].isoformat() if r['first_time'] else None,
                'last_time': r['last_time'].isoformat() if r['last_time'] else None,
                'message_count': r['message_count'],
                'preview': preview,
                'total_tokens': r['total_tokens'],
            })
        return results, total


async def delete_conversation(session_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE session_id = $1", session_id)
        await conn.execute("DELETE FROM conversation_titles WHERE session_id = $1", session_id)
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = $1", session_id)


async def batch_delete_conversations(session_ids: list):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE session_id = ANY($1)", session_ids)
        await conn.execute("DELETE FROM conversation_titles WHERE session_id = ANY($1)", session_ids)
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = ANY($1)", session_ids)


async def merge_sessions_to_target(source_ids: list, target_id: str) -> dict:
    if not source_ids:
        return {'merged_sessions': 0, 'merged_messages': 0, 'merged_token_records': 0}
    pool = await get_pool()
    async with pool.acquire() as conn:
        msg_count = await conn.fetchval("SELECT COUNT(*) FROM conversations WHERE session_id = ANY($1)", source_ids)
        await conn.execute("UPDATE conversations SET session_id = $1 WHERE session_id = ANY($2)", target_id, source_ids)
        token_count = await conn.fetchval("SELECT COUNT(*) FROM token_usage WHERE session_id = ANY($1)", source_ids)
        await conn.execute("UPDATE token_usage SET session_id = $1 WHERE session_id = ANY($2)", target_id, source_ids)
        await conn.execute("DELETE FROM conversation_titles WHERE session_id = ANY($1)", source_ids)
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = ANY($1)", source_ids)
        return {'merged_sessions': len(source_ids), 'merged_messages': msg_count or 0, 'merged_token_records': token_count or 0}


async def list_all_session_cache_states() -> list:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT scs.session_id, scs.summary, scs.a_start_round, scs.updated_at,
                   COALESCE(c.message_count, 0) as message_count,
                   COALESCE(tu.chat_tokens, 0) as chat_tokens
            FROM session_cache_state scs
            LEFT JOIN (SELECT session_id, COUNT(*) as message_count FROM conversations GROUP BY session_id) c ON scs.session_id = c.session_id
            LEFT JOIN (SELECT session_id, SUM(total_tokens) as chat_tokens FROM token_usage WHERE usage_type = 'chat' GROUP BY session_id) tu ON scs.session_id = tu.session_id
            ORDER BY scs.updated_at DESC
        """)
        results = []
        for r in rows:
            raw_summary = r['summary'] or ''
            try:
                import json
                parsed = json.loads(raw_summary)
                if isinstance(parsed, list):
                    summary_parts = parsed
                else:
                    summary_parts = [raw_summary] if raw_summary else []
            except (json.JSONDecodeError, ValueError):
                summary_parts = [raw_summary] if raw_summary else []
            results.append({
                'session_id': r['session_id'],
                'summary': '\n\n'.join(summary_parts),
                'summary_length': sum(len(p) for p in summary_parts),
                'summary_count': len(summary_parts),
                'a_start_round': r['a_start_round'],
                'updated_at': r['updated_at'].isoformat() if r['updated_at'] else None,
                'message_count': r['message_count'],
                'chat_tokens': r['chat_tokens'],
            })
        return results


async def delete_session_cache_state(session_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM session_cache_state WHERE session_id = $1", session_id)


def db_row_to_message(row: dict) -> dict:
    """
    把DB记录还原成API消息格式。
    
    普通消息: {"role": "user", "content": "你好"} 
    工具调用: {"role": "assistant", "content": null, "tool_calls": [...]}
    工具结果: {"role": "tool", "content": "结果", "tool_call_id": "call_xxx"}
    思维链:   {"role": "assistant", "content": "回答", "reasoning_content": "思维链"}
    """
    import json as _json
    msg = {"role": row["role"], "content": row.get("content") or ""}
    
    meta_str = row.get("metadata")
    if meta_str:
        try:
            meta = _json.loads(meta_str)
            # assistant 带 tool_calls
            if "tool_calls" in meta:
                msg["tool_calls"] = meta["tool_calls"]
                if not row.get("content"):
                    msg["content"] = None
            # assistant 带 reasoning_content（deepseek thinking mode）
            if "reasoning_content" in meta:
                msg["reasoning_content"] = meta["reasoning_content"]
            # tool 消息带 tool_call_id
            if "tool_call_id" in meta:
                msg["tool_call_id"] = meta["tool_call_id"]
            # 其他可能的字段（name 等）
            if "name" in meta:
                msg["name"] = meta["name"]
        except Exception:
            pass
    
    return msg


async def export_all_conversations():
    """导出所有对话记录（用于备份）"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT session_id, role, content, model, created_at
            FROM conversations
            ORDER BY session_id, created_at
        """)
        return [
            {
                'session_id': r['session_id'],
                'role': r['role'],
                'content': r['content'],
                'model': r['model'] or '',
                'created_at': r['created_at'].isoformat() if r['created_at'] else None,
            }
            for r in rows
        ]


async def import_conversations(records: list):
    """
    导入对话记录（自动去重）
    
    records: [{ session_id, role, content, model?, created_at? }, ...]
    按 session_id + role + created_at 三元组去重，已存在的跳过。
    返回 (导入数量, 跳过数量)
    """
    if not records:
        return 0, 0
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        imported = 0
        skipped = 0
        for r in records:
            session_id = r.get('session_id')
            role = r.get('role')
            content = r.get('content')
            
            if not all([session_id, role, content]):
                continue
            
            model = r.get('model', '')
            created_at = r.get('created_at')
            
            # 解析时间
            from datetime import datetime
            if created_at and isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    created_at = None
            
            # 去重检查
            if created_at:
                existing = await conn.fetchrow("""
                    SELECT id FROM conversations
                    WHERE session_id = $1 AND role = $2 AND created_at = $3
                    LIMIT 1
                """, session_id, role, created_at)
                
                if existing:
                    skipped += 1
                    continue
                
                await conn.execute("""
                    INSERT INTO conversations (session_id, role, content, model, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                """, session_id, role, content, model, created_at)
            else:
                await conn.execute("""
                    INSERT INTO conversations (session_id, role, content, model)
                    VALUES ($1, $2, $3, $4)
                """, session_id, role, content, model)
            
            imported += 1
        
        if skipped:
            print(f"📥 导入对话: {imported} 条新增, {skipped} 条已存在跳过")
        else:
            print(f"📥 导入对话: {imported} 条新增")
        
        return imported, skipped
