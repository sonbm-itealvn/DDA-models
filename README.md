# DDA-models

Mã nguồn huấn luyện mô hình Dynamic Difficulty Adjustment (DDA) cho game tower defense dựa trên telemetry.

## Mục tiêu

Mô hình chính học:

```text
P(segment_success = 1 | history_60s, context_t, candidate_difficulty)
```

Repository gồm:

- preprocessing và kiểm tra schema;
- tạo cửa sổ 12 snapshot x 5 giây;
- baseline Logistic Regression;
- baseline Random Forest nếu cài scikit-learn;
- Candidate-Conditioned GRU nếu cài PyTorch;
- calibration, metrics và model manifest;
- export artifact phục vụ Unity/ONNX ở bước tiếp theo.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Môi trường tối thiểu để chạy preprocessing:

```bash
pip install numpy pandas
```

Môi trường đầy đủ để train baseline + GRU:

```bash
pip install -r requirements.txt
```

## Dataset đầu vào cho training

Training dataset dạng CSV/Parquet cần có:

```text
session_id
player_id_hash
decision_time
segment_id
current_difficulty
candidate_difficulty
history_json        # ma trận [12, F], JSON string
context_json        # vector hoặc object JSON
segment_success     # 0/1
```

Nếu `candidate_difficulty` là số nguyên, pipeline tự chuyển thành difficulty vector từ `configs/difficulty_profiles.json`. Có thể cung cấp `difficulty_vector_json` nếu game dùng profile tùy biến.

## Chuẩn bị từ log game

```bash
python scripts/prepare_dataset.py \
  --input /path/to/session.json \
  --output data/features/dda_windows_v1.parquet
```

Script không nội suy event giả. Nếu log chỉ có snapshot thưa hoặc thiếu `segmentId`/outcome tương lai, script sẽ báo số window không tạo được thay vì gán nhãn sai.

## Train

```bash
python scripts/train.py \
  --dataset data/features/dda_windows_v1.parquet \
  --output-dir artifacts/run_v1 \
  --epochs 50
```

Kết quả gồm:

```text
metrics.json
feature_contract.json
split_manifest.json
scaler.json
logistic_model.joblib
random_forest.joblib       # nếu bật RF
candidate_gru.pt           # nếu PyTorch có mặt
model_manifest.json
```

Nếu thiếu scikit-learn hoặc PyTorch, script vẫn chạy model khả dụng và ghi rõ model nào bị bỏ qua trong `metrics.json`.

## Dataset hiện tại của dự án

`loggame.json` mẫu trước đây có thể dùng để kiểm tra ingest/schema, nhưng chưa đủ `segmentId`, `objectiveId` và outcome theo segment để train Candidate-Conditioned DDA hợp lệ. Cần logger mới ghi:

- `segment_start`, `segment_finish`, `objective_success`, `objective_failure`;
- `applied_difficulty`;
- event combat và hành vi;
- snapshot mỗi 5–10 giây;
- `player_id_hash` ổn định.

## Nguyên tắc chống leakage

- split theo `player_id_hash`, fallback theo `session_id`;
- scaler chỉ fit trên train;
- label thuộc segment kế tiếp, không dùng `result=Won` cho mọi snapshot;
- `death_in_horizon`, `damage_taken_horizon` và `segment_success` không được đưa vào input;
- chỉ dùng difficulty thực tế đã áp dụng.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

## Cấu trúc

```text
src/ddamodels/
  data.py       # ingest, validate, feature/window building
  models.py     # logistic, RF, candidate-conditioned GRU
  train.py      # split, train, metrics, artifacts
scripts/
  prepare_dataset.py
  train.py
configs/
  difficulty_profiles.json
```
