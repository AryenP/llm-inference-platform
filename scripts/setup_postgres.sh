#!/usr/bin/env bash
# Pods are containers with no docker daemon, so postgres runs natively rather than
# through docker-compose. Run once per pod; the data directory lives on the volume.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates lsb-release

install -d /usr/share/postgresql-common/pgdg
curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
  https://www.postgresql.org/media/keys/ACCC4CF8.asc
echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] \
https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
  > /etc/apt/sources.list.d/pgdg.list

apt-get update -qq
apt-get install -y -qq postgresql-17 postgresql-17-pgvector

service postgresql start
until pg_isready -q; do sleep 1; done

su postgres -c "psql -tAc \"select 1 from pg_roles where rolname='rag'\"" | grep -q 1 ||
  su postgres -c "psql -c \"create role rag login password 'rag' superuser\""
su postgres -c "psql -tAc \"select 1 from pg_database where datname='rag'\"" | grep -q 1 ||
  su postgres -c "createdb -O rag rag"

psql postgresql://rag:rag@localhost:5432/rag -f "$(dirname "$0")/../sql/001_init.sql"
echo "postgres ready: postgresql://rag:rag@localhost:5432/rag"
