-- pgvector is the production embedding store; the API degrades to JSON float arrays on
-- SQLite so demos need no database server.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
