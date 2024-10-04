echo "[i] installing dependencies with poetry"
# install project depencies with poetry
poetry install
# install pytorch-fast-transformers after so it can pick up torch
poetry run pip install pytorch-fast-transformers --no-build-isolation
echo "[i] Done."