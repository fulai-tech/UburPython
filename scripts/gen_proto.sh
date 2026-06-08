#!/usr/bin/env bash
# 由 proto/bionode_comm.proto（及依赖 bionode_common.proto）生成 gRPC stub 到 app/bionode_grpc_clients/comm/grpc_gen/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$ROOT/proto"
OUT_DIR="$ROOT/app/bionode_grpc_clients/comm/grpc_gen"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

python -m grpc_tools.protoc \
  -I"$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/bionode_common.proto" \
  "$PROTO_DIR/bionode_comm.proto"

# grpcio 生成物使用绝对 import，修正为包内相对 import
if [[ "$(uname)" == "Darwin" ]]; then
  sed -i '' 's/^import bionode_common_pb2/from . import bionode_common_pb2/' "$OUT_DIR/bionode_comm_pb2.py" 2>/dev/null || true
  sed -i '' 's/^import bionode_comm_pb2/from . import bionode_comm_pb2/' "$OUT_DIR/bionode_comm_pb2_grpc.py" 2>/dev/null || true
else
  sed -i 's/^import bionode_common_pb2/from . import bionode_common_pb2/' "$OUT_DIR/bionode_comm_pb2.py" 2>/dev/null || true
  sed -i 's/^import bionode_comm_pb2/from . import bionode_comm_pb2/' "$OUT_DIR/bionode_comm_pb2_grpc.py" 2>/dev/null || true
fi

echo "gRPC stub generated at $OUT_DIR"
