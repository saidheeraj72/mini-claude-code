-- mini-claude-code schema
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------- sessions
CREATE TABLE IF NOT EXISTS sessions (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    cwd         text        NOT NULL,
    model       text        NOT NULL,
    title       text,
    status      text        NOT NULL DEFAULT 'active',
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sessions_cwd_updated_idx
    ON sessions (cwd, updated_at DESC);

-- ---------------------------------------------------------------- messages
CREATE TABLE IF NOT EXISTS messages (
    id           bigserial PRIMARY KEY,
    session_id   uuid        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq          int         NOT NULL,
    role         text        NOT NULL,   -- user|assistant|tool|system|summary
    content      text        NOT NULL DEFAULT '',
    tool_calls   jsonb,
    tool_name    text,                   -- set when role='tool'
    token_count  int         NOT NULL DEFAULT 0,
    superseded   boolean     NOT NULL DEFAULT false,  -- folded into a summary
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, seq)
);
CREATE INDEX IF NOT EXISTS messages_session_seq_idx
    ON messages (session_id, seq);

-- -------------------------------------------------------------- tool calls
CREATE TABLE IF NOT EXISTS tool_calls (
    id           bigserial PRIMARY KEY,
    session_id   uuid        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id   bigint      REFERENCES messages(id) ON DELETE SET NULL,
    name         text        NOT NULL,
    args         jsonb       NOT NULL DEFAULT '{}'::jsonb,
    result       text,
    is_error     boolean     NOT NULL DEFAULT false,
    duration_ms  int,
    approved_by  text,                   -- auto|user|rule|yolo
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tool_calls_session_idx
    ON tool_calls (session_id, created_at DESC);

-- ------------------------------------------------------------- permissions
-- session_id NULL means the rule is global
CREATE TABLE IF NOT EXISTS permissions (
    id          bigserial PRIMARY KEY,
    session_id  uuid        REFERENCES sessions(id) ON DELETE CASCADE,
    tool_name   text        NOT NULL,
    pattern     text        NOT NULL DEFAULT '*',
    decision    text        NOT NULL,    -- allow|deny
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS permissions_unique_rule
    ON permissions (COALESCE(session_id, '00000000-0000-0000-0000-000000000000'::uuid),
                    tool_name, pattern);

-- ------------------------------------------------------------ code index
CREATE TABLE IF NOT EXISTS files (
    id          bigserial PRIMARY KEY,
    repo_root   text        NOT NULL,
    path        text        NOT NULL,
    sha256      text        NOT NULL,
    mtime       timestamptz,
    size        int,
    lang        text,
    indexed_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (repo_root, path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    file_id     bigint      NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    start_line  int         NOT NULL,
    end_line    int         NOT NULL,
    content     text        NOT NULL,
    embedding   vector(768)
);
CREATE INDEX IF NOT EXISTS chunks_file_idx ON chunks (file_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
