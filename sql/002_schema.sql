create table if not exists papers (
    arxiv_id   text primary key,
    title      text not null,
    abstract   text not null,
    categories text[] not null,
    published  timestamptz not null,
    updated    timestamptz
);

create table if not exists chunks (
    id        bigserial primary key,
    arxiv_id  text not null references papers(arxiv_id) on delete cascade,
    ord       int not null,
    text      text not null,
    embedding vector(1024),
    tsv       tsvector generated always as (to_tsvector('english', text)) stored,
    unique (arxiv_id, ord)
);

-- the tsvector exists from the start so week 5's BM25 half needs no migration
create index if not exists chunks_tsv_idx on chunks using gin (tsv);

create index if not exists chunks_embedding_idx on chunks
    using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);
