# Dockerfile

# 1. Immagine di partenza: usiamo un'immagine Python ufficiale e snella
FROM python:3.10-slim

# 2. Imposta la cartella di lavoro all'interno del container
WORKDIR /app

# 3. Copia i file necessari per l'installazione del package
# --- ECCO LA CORREZIONE: Aggiunto README.md alla lista ---
COPY setup.py requirements.txt README.md ./

# 4. Installa le dipendenze definite nel requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia il codice sorgente del tuo package
COPY src/ ./src/

# 6. Installa il tuo package (speed-analyzer) all'interno del container
RUN pip install .