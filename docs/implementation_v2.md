# Kế hoạch triển khai DDA v2

## Phạm vi nghiên cứu đã điều chỉnh

Bản v2 không tuyên bố một model duy nhất chạy nguyên xi cho mọi game. Nó cung cấp SDK contract và model backbone có thể tái sử dụng nếu game triển khai adapter semantic đúng chuẩn.

### Mô-đun đã có trong repository

- `canonical.py`: canonical signals, availability mask, session-relative normalization.
- `models.py`: gated weighted Tokenized FiLM GRU với tensor tĩnh, bên cạnh baseline và Candidate-Conditioned GRU cũ.
- `adaptation.py`: batch MAP temperature calibration causal, bounds `[0.7, 2.0]`, sliding window.
- `controller.py`: uncertainty-aware scoring + cooldown/hysteresis/safety gate.
- `segments.py`: event-bounded/fixed-horizon segment contract.
- `data.py`: `build_tokenized_window_rows()` tạo trực tiếp rows v2 từ canonical frame; `load_tokenized_window_dataset()` kiểm tra và nạp các mask.
- `train_adaptive.py`: train model tokenized từ `values_json`, `mask_json`, `difficulty_vector_json`.
- `canonical_feature_contract.json`: semantic IDs và difficulty contract.

## Dataset v2

Một row adaptive cần:

```text
values_json              # [T, 21] canonical values
mask_json                # [T, 21], 1=available, 0=missing
context_json             # context đã canonical hóa
difficulty_vector_json   # [5]: relative_step, pressure_delta, forgiveness_delta, resource_delta, empirical_prior
difficulty_mask_json     # [5], prior chỉ available từ fixed randomized same domain/patch
segment_success          # label tương lai
player_id_hash/session_id
```

Giá trị missing có thể là placeholder 0 trong tensor, nhưng `mask_json` và `difficulty_mask_json` bắt buộc phải ghi 0. Model không được hiểu missing là giá trị 0 thực.

Feature bounded `[0,1]` giữ nguyên giá trị tuyệt đối. Feature không bounded dùng causal shrinkage về game baseline, `sigma_min` và clip `[-8,8]`; không dùng IQR với denominator gần 0.

Difficulty delta dùng công thức bắt buộc:

```text
clip(log(candidate/current) / log(max_step_ratio), -1, 1)
```

Khi train, `empirical_prior` được mask-dropout mặc định 35% để model học được cả đường có prior và cold-start không có prior.

## Quy trình train v2

```bash
PYTHONPATH=src python scripts/train_adaptive.py \
  --dataset data/features/canonical_windows_v1.parquet \
  --output-dir artifacts/adaptive_v1 \
  --epochs 50
```

So sánh tối thiểu:

```text
Logistic canonical aggregate
Fixed-vector GRU
TokenizedFiLMGRU
TokenizedFiLMGRU + feature dropout evaluation
```

FiLM đặt **sau GRU**: history được mã hóa một lần, sau đó ba candidate `{-1,0,+1}` chỉ chạy qua nhánh FiLM/head nhẹ.

## Causal adaptation

- warm-up/session-relative statistics chỉ dùng quan sát trước hiện tại;
- temperature calibrator chỉ buffer outcome sau khi segment kết thúc và chỉ update sau batch 20–30 segment cùng patch/domain;
- empirical prior chỉ lấy từ fixed randomized data, zero-shot phải mask prior;
- không update backbone trong trận chính;
- adapter few-shot phải fit trên pilot session, test session giữ độc lập;
- nếu uncertainty cao, controller HOLD.

## Phân tích khả thi

### Nên làm trong đề tài chính

1. Canonical adapter cho Tower Defense.
2. Cross-map hoặc cross-patch split.
3. Missing-feature robustness.
4. Calibration và uncertainty.
5. Safe controller.

### Chỉ làm nếu còn thời gian

1. Mini-game thứ hai.
2. Few-shot output adapter.
3. So sánh exploratory cross-genre.

### Để future work

1. Contextual bandit làm policy chính.
2. CQL/offline RL.
3. LoRA online trên backbone.
4. Zero-shot arbitrary game.

## Tiêu chí thành công

Model v2 chỉ được xem là có giá trị nếu ít nhất một trong các tiêu chí sau được chứng minh trên test độc lập:

- Brier/ECE tốt hơn static baseline trên Map2/Patch2;
- false increase rate thấp hơn khi feature dropout;
- uncertainty tăng khi thiếu feature và safety gate giảm hành động rủi ro;
- completion/quit/fairness cải thiện trong online pilot.

## Segment contract

Adapter phải khai báo một trong hai mode:

```text
event_bounded: wave/room/checkpoint, semantics 30–60 giây
fixed_horizon: game liên tục, survive_window theo 30 giây hoặc horizon đã định nghĩa
```

Không so sánh trực tiếp segment 15 giây của một game với segment 90 giây của game khác nếu chưa chuẩn hóa horizon và label definition.

Không dùng accuracy hoặc completion rate đơn độc để tuyên bố thành công.


## Phụ lục hardening v3

Các mô tả cũ về temperature-only calibration, `nn.GRU`, opset 17 và FiLM `gamma*h+beta` đã được thay bằng implementation hiện tại: affine logit MAP `(beta, bias)`, static unrolled GRU, export opset 15 và FiLM identity-preserving `(1+delta_gamma)*h+beta`.

`data.py` có `causal_normalize_canonical_frame()` để bảo đảm thứ tự transform trước update và giữ NaN/mask; `models.py` có dropout, empirical-prior dropout, feature-group dropout, validation và early stopping; `train_adaptive.py` ghi test metrics gồm failure PR-AUC và Brier decomposition; `controller.py` có input validation, Score direction, `dt_seconds` và `max_difficulty`.

Unit test code-level đã pass trong môi trường không có PyTorch. Neural training, ONNX Runtime parity và Unity Sentis parity vẫn là hạng mục nghiệm thu riêng, không được đánh dấu hoàn tất chỉ từ compile/unit test preprocessing.
