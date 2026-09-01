import os

# settings has no default model — it is host-specific and must fail loudly in
# production. Tests never reach vLLM, so any path satisfies it.
os.environ.setdefault("MODEL", "/models/test")
