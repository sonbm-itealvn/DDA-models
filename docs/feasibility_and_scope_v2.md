# Đánh giá tính khả thi và phạm vi DDA v2

## 1. Kết luận điều hành

Ghi chú nghiên cứu đã phát hiện đúng 5 điểm nghẽn của bản prototype:

1. phụ thuộc vào schema Tower Defense;
2. difficulty vector đóng;
3. covariate/concept shift khi đổi map hoặc patch;
4. offline model và rule controller thiếu thích nghi;
5. novelty/generalizability còn yếu nếu chỉ trình bày GRU trên một game.

Tuy nhiên, tuyên bố “một model dùng nguyên xi cho mọi game” là quá mạnh. Mỗi game có semantics, objective và actuator khác nhau. Giải pháp có tính khả thi là:

```text
Universal runtime contract
+ canonical human/game signals
+ game adapter
+ masked/tokenized temporal model
+ session-relative normalization
+ calibrated uncertainty
+ safe controller
```

Không hứa hẹn zero-shot cross-genre ở mức sản phẩm. Mục tiêu nghiên cứu khả thi trong một học kỳ là chứng minh generalization theo ba bậc:

```text
Bậc 1: cùng game, khác session/player
Bậc 2: khác map/patch, few-shot calibration
Bậc 3: khác mini-game, exploratory transfer
```

## 2. Đánh giá từng hướng xử lý

| Vấn đề | Hướng đề xuất | Khả thi | Quyết định |
|---|---|---:|---|
| Feature Tower Defense bị khóa cứng | Canonical semantic adapter | Cao | Triển khai ngay |
| F cố định và missing feature | Tokenization + mask + temporal backbone | Trung bình–cao | Triển khai bản nhỏ, so với baseline |
| Difficulty vector 5 số | Relative perturbation + empirical prior | Cao | Triển khai ngay |
| FiLM conditioning | Điều biến hidden state bằng difficulty | Trung bình | Triển khai trong model nâng cấp |
| Map/patch shift | Running/session-relative normalization | Cao | Triển khai ngay |
| Transfer learning | Frozen backbone + lightweight adapter | Trung bình | Chỉ làm few-shot, không cam kết zero-shot |
| Online temperature scaling | Cập nhật calibrator, không cập nhật backbone | Cao | Triển khai an toàn |
| Residual/LoRA adapter | Fine-tune online | Trung bình–thấp | Offline pilot trước, không cập nhật tự do trong trận |
| Contextual bandit | Controller học policy | Trung bình–thấp | Không dùng làm controller chính trong MVP |
| Offline RL/CQL | Học policy từ log | Thấp với dataset hiện tại | Để future work |
| Cross-map/cross-patch | Benchmark chính | Cao | Bắt buộc nếu có dữ liệu |
| Cross-genre | Mini-game thứ hai | Thấp–trung bình | Exploratory, không phải tiêu chí thành công chính |

## 3. Kiến trúc v2 được chốt

```text
Game-specific telemetry
        |
        v
IGameTelemetryAdapter
        |
        v
Canonical feature contract
  - progress/tempo
  - vitality/error budget
  - capability/economy
  - human cognitive-motor
        |
        v
Session-relative normalizer + missing mask
        |
        v
Feature-tokenized temporal encoder
        |
        v
FiLM difficulty conditioning
        |
        v
p_success(candidate difficulty)
        |
        +--> calibration + uncertainty
        |
        v
Safe uncertainty-aware controller
        |
        v
IDifficultyActuator tại safe checkpoint
```

## 4. Canonical feature contract

### 4.1. Nhóm Progress/Tempo

| Canonical signal | Khoảng/đơn vị | Tower Defense mapping | Game khác |
|---|---|---|---|
| `progress_ratio` | `[0,1]` | wave/segment progress | distance, objective completion |
| `progress_velocity` | ratio/s | tốc độ hoàn thành wave | tốc độ chạy/giải puzzle |
| `time_budget_ratio` | `[0,1+]` | elapsed/expected duration | timer usage |
| `tempo_rate` | actions/s hoặc event/s | spawn/action tempo | fire rate, input tempo |

### 4.2. Nhóm Vitality/Error Budget

| Canonical signal | Ý nghĩa |
|---|---|
| `vitality_ratio` | HP/giáp/mạng còn lại so với tối đa |
| `vitality_loss_rate` | tốc độ mất HP/giáp/mạng |
| `critical_error_rate` | lỗi nghiêm trọng trong cửa sổ |
| `failure_cost_ratio` | tiến độ/tài nguyên mất sau lỗi |

### 4.3. Nhóm Capability/Economy

| Canonical signal | Ý nghĩa |
|---|---|
| `resource_reserve_ratio` | tài nguyên dự trữ so với chi phí hành động chuẩn |
| `resource_utilization_rate` | tốc độ biến tài nguyên thành capability |
| `capability_ratio` | sức mạnh hiện tại so với yêu cầu segment |
| `threat_pressure_ratio` | áp lực đe dọa tương đối |

### 4.4. Nhóm Human Cognitive-Motor

| Canonical signal | Ý nghĩa |
|---|---|
| `action_frequency` | số meaningful action/giây |
| `invalid_action_rate` | tỷ lệ action vô hiệu |
| `idle_ratio` | tỷ lệ thời gian không có meaningful action |
| `reaction_latency` | độ trễ trước event cần phản ứng |
| `error_slope` | lỗi tăng hay giảm theo thời gian |
| `recent_segment_win_rate` | success rate gần đây |
| `efficiency_ratio` | output hữu ích trên input/action |

Các signal này không tự động có trong mọi game. Adapter phải trả về `value` và `available_mask`. Missing feature không được thay bằng 0 mà không có mask.

## 5. Difficulty representation v2

Thay vì bắt model hiểu `enemy_count_ratio` của Tower Defense, difficulty candidate gồm:

```text
relative_step       # -1, 0, +1 hoặc perturbation liên tục
pressure_delta      # thay đổi áp lực dự kiến
forgiveness_delta   # thay đổi error budget/assist
resource_delta      # thay đổi supply/recovery
empirical_prior     # success prior lịch sử của candidate
candidate_mask      # candidate có hợp lệ không
```

`pressure_delta` và các thành phần khác do game adapter chuẩn hóa. Controller chỉ được xét candidate mà actuator đã kiểm tra playability.

## 6. Model v2

### 6.1. Feature-tokenized temporal encoder

Mỗi timestep có các token:

```text
(feature_embedding_i, normalized_value_i, availability_mask_i)
```

Feature encoder dùng masked attention/pooling để tạo latent state `z_t` cố định chiều. Vì vậy model vẫn nhận cùng chiều latent dù game thiếu một số feature.

Điều này không có nghĩa model tự hiểu feature hoàn toàn mới. Feature mới phải có semantic ID nằm trong canonical vocabulary hoặc cần adapter/fine-tune embedding.

### 6.2. FiLM conditioning

Difficulty embedding sinh ra:

```text
gamma = Gamma(difficulty)
beta  = Beta(difficulty)
h'_t = gamma * h_t + beta
```

Sau đó GRU/temporal head dự đoán `p_success`.

### 6.3. Baseline vẫn bắt buộc

So sánh:

```text
Majority
Logistic Regression trên canonical aggregate
Random Forest
Fixed-vector GRU
Tokenized-FiLM GRU
```

Nếu tokenized model không vượt baseline trong cross-map/cross-patch, giữ baseline để làm model triển khai. Novelty phải được chứng minh bằng robustness/calibration, không chỉ bằng kiến trúc phức tạp.

## 7. Normalization và adaptation

### 7.1. Session-relative normalization

Dùng baseline trong warm-up hoặc rolling window:

```text
x_norm = (x - running_median) / (running_iqr + epsilon)
```

Hoặc tỷ lệ theo reference:

```text
x_relative = x / (reference_value + epsilon)
```

Không cập nhật thống kê bằng label tương lai. Trong 30 giây đầu, dùng profile baseline của game nếu chưa đủ warm-up.

### 7.2. Online calibration

Cho phép cập nhật temperature/platt calibrator bằng outcome đã quan sát ở các segment trước. Không cập nhật backbone trong trận chính.

Quy tắc:

```text
calibrator update -> được phép sau segment finish
backbone update   -> chỉ offline/few-shot pilot
controller update -> bị safety gate giới hạn
```

### 7.3. Lightweight game adapter

Adapter có thể là:

- mapping game-specific -> canonical signals;
- affine residual ở output;
- calibration temperature;
- embedding của objective type.

Few-shot adaptation cần đánh giá bằng leave-session-out. Không được dùng dữ liệu của chính test session để tinh chỉnh rồi báo cáo như zero-shot.

## 8. Controller v2

Contextual bandit/CQL chưa phù hợp làm lõi MVP vì cần nhiều action được khám phá, logging policy và dữ liệu phản thực. Controller v2 dùng:

```text
candidate score
+ calibrated success probability
+ uncertainty penalty
+ change-cost penalty
+ struggle safety gate
+ cooldown/hysteresis
```

Công thức:

```text
score(d) =
  -abs(p_success(d) - p_target)
  - lambda_uncertainty * uncertainty(d)
  - lambda_change * abs(relative_step(d))
```

Controller chỉ chọn trong candidate hợp lệ `{-1,0,+1}` và chỉ áp dụng ở safe checkpoint.

Bandit có thể là thí nghiệm phụ sau khi có:

- policy log đầy đủ;
- giới hạn exploration;
- reward định nghĩa trước;
- offline policy evaluation;
- safety constraint không thể vượt qua.

## 9. Thiết kế thực nghiệm khả thi

### Thực nghiệm A — Cross-map/cross-patch, bắt buộc

- Train trên Map1/patch v1.
- Test zero-shot trên Map2/patch v2.
- So sánh static scaler + fixed GRU với canonical + dynamic normalizer + adapter calibration.
- Đo PR-AUC, Brier, ECE, false increase rate, oscillation và completion.

### Thực nghiệm B — Feature dropout, bắt buộc

Che từng nhóm:

```text
capability/economy
vitality/error
human cognitive-motor
progress/tempo
```

Đo:

- degradation của PR-AUC/Brier;
- uncertainty tăng đúng khi thiếu tín hiệu;
- safety gate có chuyển sang HOLD hay không.

### Thực nghiệm C — Cross-genre, tùy chọn

Dựng mini-game thứ hai. Chỉ tuyên bố exploratory transfer nếu:

- canonical adapter có semantics rõ;
- có ít nhất một baseline cùng điều kiện;
- test session chưa dùng trong adapter fitting;
- báo cáo riêng zero-shot và few-shot.

## 10. Những tuyên bố không nên đưa vào luận văn

Không nên tuyên bố:

- model dùng được nguyên xi cho mọi game;
- tokenization tự hiểu feature hoàn toàn mới không cần adapter;
- online learning luôn cải thiện trải nghiệm;
- bandit tốt hơn rule controller khi chưa có policy evaluation;
- cross-genre transfer đã được chứng minh chỉ bằng mô phỏng feature dropout.

Nên tuyên bố:

> Đề tài đề xuất một runtime DDA có canonical telemetry contract, masked temporal representation, session-relative normalization và safe uncertainty-aware controller. Tính khái quát được đánh giá trên cross-map/cross-patch; cross-genre là kiểm chứng thăm dò có adapter.

## 11. Database audit contract

Mọi bảng database phải có:

```text
create_at DATETIME WITH TIME ZONE
create_by DATETIME WITH TIME ZONE
update_at DATETIME WITH TIME ZONE
update_by DATETIME WITH TIME ZONE
```

Nên bổ sung `created_by_actor` và `updated_by_actor` nếu cần biết service/user tạo hoặc sửa bản ghi, vì `create_by`/`update_by` theo quy ước hiện tại là timestamp.
