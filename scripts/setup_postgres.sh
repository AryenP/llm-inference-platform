#!/usr/bin/env bash
# Postgres + pgvector without docker: RunPod pods have no docker daemon, and WSL
# doesn't either unless you go out of your way. Re-execs itself under sudo.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || exec sudo -E "$0" "$@"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq

pgvector_version() {
  apt-cache search --names-only '^postgresql-[0-9]+-pgvector$' |
    grep -oE 'postgresql-[0-9]+' | grep -oE '[0-9]+' | sort -rn | head -1
}

# the distro often carries pgvector already; only reach for PGDG when it doesn't
ver=$(pgvector_version || true)
if [ -z "$ver" ]; then
  apt-get install -y -qq curl ca-certificates lsb-release
  install -d /usr/share/postgresql-common/pgdg
  curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
    https://www.postgresql.org/media/keys/ACCC4CF8.asc
  echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
    > /etc/apt/sources.list.d/pgdg.list
  apt-get update -qq
  ver=$(pgvector_version || true)
fi

if [ -z "$ver" ]; then
  echo "no postgresql-*-pgvector package available for $(lsb_release -cs 2>/dev/null || echo this release)" >&2
  echo "PGDG may not publish for it yet — check https://apt.postgresql.org/pub/repos/apt/dists/" >&2
  exit 1
fi

apt-get install -y -qq "postgresql-$ver" "postgresql-$ver-pgvector"

service postgresql start
until pg_isready -q; do sleep 1; done

su postgres -c "psql -tAc \"select 1 from pg_roles where rolname='rag'\"" | grep -q 1 ||
  su postgres -c "psql -c \"create role rag login password 'rag' superuser\""
su postgres -c "psql -tAc \"select 1 from pg_database where datname='rag'\"" | grep -q 1 ||
  su postgres -c "createdb -O rag rag"

psql postgresql://rag:rag@localhost:5432/rag -f "$(dirname "$0")/../sql/001_init.sql"
echo "postgres $ver ready: postgresql://rag:rag@localhost:5432/rag"
