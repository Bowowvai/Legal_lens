/* --------------------------------------------------------------
   1️⃣ Enable the pgcrypto extension for UUID generation
   -------------------------------------------------------------- */
create extension if not exists pgcrypto;

/* --------------------------------------------------------------
   2️⃣ Create a *private* bucket named `documents`
   Note: Some projects don't expose storage.create_bucket();
   inserting into storage.buckets is reliable and idempotent.
   -------------------------------------------------------------- */
insert into storage.buckets (id, name, public)
values ('documents'::text, 'documents', false)
on conflict (id) do nothing;   -- idempotent

/* --------------------------------------------------------------
   3️⃣ Table that stores metadata for each uploaded file
   -------------------------------------------------------------- */
create table if not exists public.documents (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users(id) on delete cascade,
  title      text not null,
  file_path  text not null,
  mime_type  text,
  size       bigint,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

/* --------------------------------------------------------------
   Enable Row‑Level Security for the table (once)
   -------------------------------------------------------------- */
alter table public.documents enable row level security;

/* --------------------------------------------------------------
   4️⃣ Keep `updated_at` fresh – create function only once
   -------------------------------------------------------------- */
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

/* --------------------------------------------------------------
   Drop the trigger if it already exists, then create it
   -------------------------------------------------------------- */
drop trigger if exists set_documents_updated_at on public.documents;

create trigger set_documents_updated_at
before update on public.documents
for each row execute function public.set_updated_at();

/* --------------------------------------------------------------
   5️⃣ RLS policies – users can only work with their own rows
   -------------------------------------------------------------- */
-- Drop existing policies if they exist (Postgres has no IF NOT EXISTS for policies)
drop policy if exists documents_select_own on public.documents;
create policy "documents_select_own" on public.documents
  for select to authenticated
  using (auth.uid() = user_id);

drop policy if exists documents_insert_own on public.documents;
create policy "documents_insert_own" on public.documents
  for insert to authenticated
  with check (auth.uid() = user_id);

drop policy if exists documents_update_own on public.documents;
create policy "documents_update_own" on public.documents
  for update to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists documents_delete_own on public.documents;
create policy "documents_delete_own" on public.documents
  for delete to authenticated
  using (auth.uid() = user_id);

/* --------------------------------------------------------------
   6️⃣ Helpful index for the common access pattern
   -------------------------------------------------------------- */
create index if not exists documents_user_id_created_at_idx
  on public.documents(user_id, created_at desc);

/* --------------------------------------------------------------
   7️⃣ Storage RLS policies – each user can only read/write files
       in their own folder (`${auth.uid()}/<filename>`)
   -------------------------------------------------------------- */
drop policy if exists storage_documents_read_own on storage.objects;
create policy "storage_documents_read_own" on storage.objects
  for select to authenticated
  using (
    bucket_id = 'documents' and
    auth.uid()::text = split_part(name, '/', 1)
  );

drop policy if exists storage_documents_insert_own on storage.objects;
create policy "storage_documents_insert_own" on storage.objects
  for insert to authenticated
  with check (
    bucket_id = 'documents' and
    auth.uid()::text = split_part(name, '/', 1)
  );

drop policy if exists storage_documents_update_own on storage.objects;
create policy "storage_documents_update_own" on storage.objects
  for update to authenticated
  using (
    bucket_id = 'documents' and
    auth.uid()::text = split_part(name, '/', 1)
  )
  with check (
    bucket_id = 'documents' and
    auth.uid()::text = split_part(name, '/', 1)
  );

drop policy if exists storage_documents_delete_own on storage.objects;
create policy "storage_documents_delete_own" on storage.objects
  for delete to authenticated
  using (
    bucket_id = 'documents' and
    auth.uid()::text = split_part(name, '/', 1)
  );

/* --------------------------------------------------------------
   8️⃣ Summaries table – stores auto-saved text + generated summaries
   -------------------------------------------------------------- */
create table if not exists public.summaries (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  title       text not null,
  source_text text,
  summary     text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- Enable RLS
alter table public.summaries enable row level security;

-- Reuse updated_at trigger function; create trigger for summaries
drop trigger if exists set_summaries_updated_at on public.summaries;
create trigger set_summaries_updated_at
before update on public.summaries
for each row execute function public.set_updated_at();

-- RLS policies: users can only access their own rows
drop policy if exists summaries_select_own on public.summaries;
create policy "summaries_select_own" on public.summaries
  for select to authenticated
  using (auth.uid() = user_id);

drop policy if exists summaries_insert_own on public.summaries;
create policy "summaries_insert_own" on public.summaries
  for insert to authenticated
  with check (auth.uid() = user_id);

drop policy if exists summaries_update_own on public.summaries;
create policy "summaries_update_own" on public.summaries
  for update to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists summaries_delete_own on public.summaries;
create policy "summaries_delete_own" on public.summaries
  for delete to authenticated
  using (auth.uid() = user_id);

-- Helpful index
create index if not exists summaries_user_id_created_at_idx
  on public.summaries(user_id, created_at desc);

/* --------------------------------------------------------------
   9️⃣ Prevent duplicate summaries per user via content hash
   -------------------------------------------------------------- */
-- Add hash column
alter table public.summaries
  add column if not exists summary_hash text;

-- Compute SHA-256 hash of summary text before insert/update
create or replace function public.set_summary_hash()
returns trigger
language plpgsql
as $$
begin
  new.summary_hash := encode(digest(coalesce(new.summary, ''), 'sha256'), 'hex');
  return new;
end;
$$;

drop trigger if exists set_summaries_hash on public.summaries;
create trigger set_summaries_hash
before insert or update of summary on public.summaries
for each row execute function public.set_summary_hash();

-- Enforce uniqueness per user
create unique index if not exists summaries_user_hash_unique
  on public.summaries(user_id, summary_hash);