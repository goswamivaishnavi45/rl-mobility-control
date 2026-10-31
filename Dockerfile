FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
# Default: run the test suite, then evaluate the shipped models against PID/LQR
CMD ["sh", "-c", "pytest -q tests && python scripts/evaluate.py --episodes 100"]
