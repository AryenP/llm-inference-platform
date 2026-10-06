#!/usr/bin/env bash
# Postgres + pgvector without docker: RunPod pods have no docker daemon, and WSL
# doesn't either unless you go out of your way. Re-execs itself under sudo.
#
# On a pod, only the network volume survives a stop — apt packages and anything
# outside it are wiped. So the binaries get reinstalled each session (cheap) but
# the data directory lives on the volume. Re-running this after a restart
# re-attaches to the existing cluster rather than building a new one.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || exec sudo -E "$0" "$@"

# a pod mounts the volume at /workspace; anywhere else, keep the distro default
if [ -z "${PGDATA:-}" ]; then
  if [ -d /workspace ]; then PGDATA=/workspace/pgdata; else PGDATA=""; fi
fi

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
  echo "no postgresql-*-pgvector package for $(lsb_release -cs 2>/dev/null || echo this release)" >&2
  echo "check https://apt.postgresql.org/pub/repos/apt/dists/ for a matching codename" >&2
  exit 1
fi

apt-get install -y -qq "postgresql-$ver" "postgresql-$ver-pgvector"

if [ -n "$PGDATA" ]; then
  bin="/usr/lib/postgresql/$ver/bin"
  # the package starts a cluster in the default location; it would hold 5432
  pg_dropcluster --stop "$ver" main 2>/dev/null || true

  # the persistent mount comes back with group/world bits set after a pod stop,
  # and postgres refuses to start on anything looser than 0750
  install -d -o postgres -g postgres -m 700 "$PGDATA"
  chown -R postgres:postgres "$PGDATA"
  chmod 700 "$PGDATA"
  if [ ! -f "$PGDATA/PG_VERSION" ]; then
    su postgres -c "$bin/initdb -D '$PGDATA' -E UTF8"
  fi
  su postgres -c "$bin/pg_ctl -D '$PGDATA' -l '$PGDATA/server.log' -o '-c listen_addresses=localhost' -w start" ||
    { tail -20 "$PGDATA/server.log" >&2; exit 1; }
else
  service postgresql start
fi

until pg_isready -q; do sleep 1; done

su postgres -c "psql -tAc \"select 1 from pg_roles where rolname='rag'\"" | grep -q 1 ||
  su postgres -c "psql -c \"create role rag login password 'rag' superuser\""
su postgres -c "psql -tAc \"select 1 from pg_database where datname='rag'\"" | grep -q 1 ||
  su postgres -c "createdb -O rag rag"

psql postgresql://rag:rag@localhost:5432/rag -f "$(dirname "$0")/../sql/001_init.sql"

echo "postgres $ver ready at ${PGDATA:-distro default}"
psql postgresql://rag:rag@localhost:5432/rag -tAc \
  "select 'papers=' || (select count(*) from papers) || ' chunks=' || (select count(*) from chunks)" \
  2>/dev/null || echo "schema not created yet — run ./init.sh ingest"
