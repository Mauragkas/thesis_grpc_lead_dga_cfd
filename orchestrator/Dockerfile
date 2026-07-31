FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir grpcio grpcio-tools
COPY eval.proto .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. eval.proto
COPY main.py .
CMD ["python", "main.py"]
