FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends openbabel curl wget && rm -rf /var/lib/apt/lists/*

WORKDIR /nexusmd

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN wget -q https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 -O /usr/local/bin/vina && chmod +x /usr/local/bin/vina && echo "Vina installed" || echo "WARNING: Vina not found"

COPY app/ ./app/
COPY frontend/ ./frontend/
RUN mkdir -p data/results data/pdb_cache data/ligands logs

ENV PYTHONPATH=/nexusmd

CMD python -c "import os,subprocess; subprocess.run(['uvicorn','app.main:app','--host','0.0.0.0','--port',os.environ.get('PORT','8000'),'--workers','1'])"
