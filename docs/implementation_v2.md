# Kế hoạch triển khai DDA v2

## Phạm vi nghiên cứu đã điều chỉnh

Bản v2 không tuyên bố một model duy nhất chạy nguyên xi cho mọi game. Nó cung cấp SDK contract và model backbone có thể tái sử dụng nếu game triển khai adapter semantic đúng chuẩn.

### Mô-đun đã có trong repository

- `canonical.py`: canonical signals, availability mask, session-relative normalization.
- `models.py`: `TokenizedFiLMGRU`, bên cạnh baseline và Candidate-Conditioned GRU cũ.
- `adaptation.py`: online temperature calibration causal.
- `controller.py`: uncertainty-aware scoring + cooldown/hysteresis/safety gate.
- `train_adaptive.py`: train model tokenized từ `values_json`, `mask_json`, `difficulty_vector_json`.
- `canonical_feature_contract.json`: semantic IDs và difficulty contract.

## Dataset v2

Một row adaptive cần:

```text
values_json              # [T, 19] canonical values
mask_json                # [T, 19], 1=available, 0=missing
context_json             # context đã canonical hóa
 difficulty_vector_json  # [5]: relative_step, pressure_delta, forgiveness_delta, resource_delta, empirical_prior
segment_success          # label tương lai
player_id_hash/session_id
```

Giá trị missing có thể là placeholder 0 trong tensor, nhưng `mask_json` bắt buộc phải ghi 0. Model không được hiểu missing là giá trị 0 thực.

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

## Causal adaptation

- warm-up/session-relative statistics chỉ dùng quan sát trước hiện tại;
- temperature calibrator chỉ update sau khi segment kết thúc;
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

Không dùng accuracy hoặc completion rate đơn độc để tuyên bố thành công.
