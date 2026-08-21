#!/usr/bin/env bash
# 由 proto/uburnode.proto、proto/uburnode_somni.proto 生成 stub → app/uburnode_grpc/grpc_gen/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$ROOT/proto"
OUT_DIR="$ROOT/app/uburnode_grpc/grpc_gen"

mkdir -p "$OUT_DIR"
touch "$ROOT/app/uburnode_grpc/__init__.py"
touch "$OUT_DIR/__init__.py"

# 清理旧分拆 stub，避免混用
rm -f "$OUT_DIR"/uburnode_audio_pb2*.py \
  "$OUT_DIR"/uburnode_quiz_pb2*.py \
  "$OUT_DIR"/bionode_common_pb2*.py

python -m grpc_tools.protoc \
  -I"$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/uburnode.proto" \
  "$PROTO_DIR/uburnode_somni.proto"

if [[ "$(uname)" == "Darwin" ]]; then
  SED_INPLACE=(sed -i '')
else
  SED_INPLACE=(sed -i)
fi

"${SED_INPLACE[@]}" \
  's/^import uburnode_pb2 as uburnode__pb2/from . import uburnode_pb2 as uburnode__pb2/' \
  "$OUT_DIR/uburnode_pb2_grpc.py"
"${SED_INPLACE[@]}" \
  's/^import uburnode_somni_pb2 as uburnode__somni__pb2/from . import uburnode_somni_pb2 as uburnode__somni__pb2/' \
  "$OUT_DIR/uburnode_somni_pb2_grpc.py"

echo "UburNode gRPC stub generated at $OUT_DIR"
ls -la "$OUT_DIR"
