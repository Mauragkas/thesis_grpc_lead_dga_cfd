FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir grpcio grpcio-tools numpy scipy matplotlib aerosandbox
COPY eval.proto .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. eval.proto
COPY main.py .
CMD ["python", "main.py"]
