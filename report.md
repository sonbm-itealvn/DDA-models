# CƠ SỞ LÝ THUYẾT, ĐỊNH NGHĨA VÀ KIẾN THỨC CHUYÊN SÂU CỦA DDA-models v2

**Bản hiệu chỉnh hoàn chỉnh: đối chiếu với kiến trúc `DDA-models (Universal Runtime Contract & Adaptive Neural Framework)`**

---

## PHẦN 0: HƯỚNG DẪN ĐỌC, QUY ƯỚC VÀ NHẬT KÝ HIỆU CHỈNH

### 0.1. Mục đích và phạm vi

Tài liệu hệ thống hóa nền tảng lý thuyết, định nghĩa và toán học của DDA-models v2, dùng làm chương *Cơ sở lý thuyết & Phương pháp nghiên cứu* của báo cáo NCKH. Mỗi khái niệm được trình bày theo ba lớp: **(1) bản chất lý thuyết**, **(2) công thức/cơ chế trong hệ thống**, **(3) giới hạn và điểm cần kiểm chứng** (dùng để trả lời phản biện).

Nguyên tắc biên soạn: tài liệu **tách bạch lý thuyết đúng khỏi hiện trạng triển khai**. Khi hiện trạng mã nguồn lệch khỏi lý thuyết đúng, tài liệu nêu cả hai và gắn nhãn để bạn biết chỗ nào cần chỉnh trong mô hình.

### 0.2. Nhãn trạng thái

| Nhãn | Ý nghĩa |
|---|---|
| **[Đã triển khai v2]** | Có trong tài liệu kiến trúc v2 (mô-đun, công thức, tham số đã nêu). |
| **[Khuyến nghị]** | Chưa có hoặc chưa rõ trong v2, nhưng cần có để mô hình đúng về lý thuyết/toán học. |
| **[Mở rộng]** | Hướng nghiên cứu mở rộng, không thuộc phạm vi cam kết của v2. |
| **[Cần xác minh]** | Chi tiết chưa được tài liệu kiến trúc nêu; cần đối chiếu mã nguồn trước khi đưa vào báo cáo. |

### 0.3. Ký hiệu toán học

| Ký hiệu | Ý nghĩa |
|---|---|
| $s$, $u(s)$ | Phiên chơi và người chơi sở hữu phiên đó |
| $k$, $t_k$ | Chỉ số phân đoạn (segment) và thời điểm quyết định (safe checkpoint) trước phân đoạn $k$ |
| $\Delta = 5\,\text{s}$ | Chu kỳ lấy mẫu snapshot |
| $L = 12$ | Độ dài cửa sổ (số snapshot); mỗi snapshot đại diện cho 5 s kết thúc tại nhãn thời gian của nó, nên cửa sổ bao phủ 60 s. *(Mã nguồn dùng $T$ cho cả độ dài chuỗi và nhiệt độ; tài liệu này dùng $L$ cho độ dài chuỗi và $\tau$ cho nhiệt độ.)* |
| $F = 21$ | Số tín hiệu canonical |
| $x_t \in \mathbb{R}^{F}$, $m_t \in \{0,1\}^{F}$ | Vector giá trị và mặt nạ khả dụng tại snapshot $t$ |
| $\mathcal{H}_t = \{(x_{t-j\Delta}, m_{t-j\Delta})\}_{j=0}^{L-1}$ | Lịch sử 12 snapshot kết thúc tại $t$ |
| $c_t \in \mathbb{R}^{C}$ | Vector ngữ cảnh ($C = 3$: `progress_ratio`, `normalized_segment_index`, `current_difficulty`) |
| $d \in \mathbb{R}^{10}$ | Vector độ khó ứng viên (5 giá trị + 5 mask) |
| $\mathsf{D}\in\{0,\dots,D_{\max}\}$ | Chỉ số profile độ khó (rời rạc); $D_{\max}$ là cấu hình của game *[Cần xác minh]* |
| $Y_k \in \{0,1\}$ | Kết quả phân đoạn $k$ (`segment_success`) |
| $z = f_\theta(\mathcal{H}, c, d)$, $p = \sigma(z)$ | Logit thô và xác suất thô của mạng |
| $\tau$, $\beta = 1/\tau$ | Nhiệt độ hiệu chuẩn và nghịch đảo của nó; $\hat p = \sigma(z/\tau) = \sigma(\beta z)$ |
| $P_{\text{target}} = 0.70$ | Xác suất thành công mục tiêu |
| $\sigma(u) = 1/(1+e^{-u})$ | Hàm sigmoid |

### 0.4. Nhật ký hiệu chỉnh so với bản tổng hợp trước

| # | Vị trí cũ | Vấn đề | Hiệu chỉnh trong bản này |
|---|---|---|---|
| 1 | 1.1 | Gọi là "phản thực" (counterfactual) nhưng chưa nêu điều kiện nhận dạng | Phát biểu lại là dự báo can thiệp có điều kiện và nêu 3 giả thiết nhận dạng (mục 1.2) |
| 2 | 1.4 | Nhóm "Human Cognitive-Motor" thực chất gồm 9 biến, trong đó có biến lịch sử/hiệu suất | Bảng 21 tín hiệu chia 4/4/4/9; chỉ 5 biến thao tác thuần mới là cơ sở cho khả năng chuyển giao (mục 2.1) |
| 3 | 2.3 | Ví dụ Simpson thực chất là confounding; thiếu positivity | Viết lại thành confounding/positivity/randomization (mục 3.3) |
| 4 | 2.5 | Ví dụ "−5σ" mâu thuẫn $\sigma_{\min}=0.05$; cận sàn tuyệt đối vi phạm thứ nguyên | Sửa ví dụ; đề xuất sàn theo từng đặc trưng (mục 2.6) |
| 5 | 3.2 | Thiếu phương trình GRU; sai quy ước tham số | Viết theo quy ước PyTorch, nêu số tham số (mục 4.2) |
| 6 | 3.4 | Khẳng định cận $10^{-6}$ "giữ tổng trọng số = 1" không chính xác | Chứng minh lại: cận chỉ có tác dụng khi tất cả bị mask; nêu mất thông tin số lượng (mục 4.3) |
| 7 | 3.5 | "Đảm bảo nhân quả" dùng sai thuật ngữ; thiếu cảnh báo mạng bỏ qua $d$ | Đổi thành nhất quán thời gian; thêm kiểm tra độ nhạy theo $d$ (mục 4.4) |
| 8 | 3.6 | Tiêu đề có Adam nhưng không giải thích | Bổ sung Adam/AdamW, so sánh L2 với weight decay (mục 4.5) |
| 9 | 4.1 | PR-AUC chưa nêu lớp dương; Reliability Diagram diễn giải sai | Lớp dương là *failure*; diễn giải lại (mục 5.1) |
| 10 | 4.2 | Temperature Scaling không sửa được lệch (bias); logit tập trung quanh $\ln(7/3)$ làm $\tau$ và bias không tách được | Chứng minh, nêu ví dụ số, đề xuất Platt/MAP có hệ số chặn (mục 5.2) |
| 11 | 4.3 | Gọi độ phủ mặt nạ là epistemic uncertainty | Đổi thành *độ tin cậy do độ phủ đầu vào*; nêu công thức đầy đủ và bảng ngưỡng (mục 5.3) |
| 12 | 8.2 | Class weight làm lệch xác suất | Công thức hiệu chỉnh $z_{\text{true}} = z - \ln(w_1/w_0)$; mặc định không weight (mục 4.6) |
| 13 | 5.2 | "Hysteresis" thực chất là debouncing | Tách dead band, debounce, lockout (mục 6.4) |
| 14 | 8.5, 8.8 | $\mathcal{U}(d)$ không phụ thuộc $d$ nên không ảnh hưởng `argmax`; entropy kém phân biệt ở $p\approx 0.7$ | Phân tích và đề xuất (mục 5.3, 6.2) |
| 15 | 6.3 | $P_{\text{actual}}$ nhị phân; "False Increase" thiếu định nghĩa struggle | Định nghĩa lại bằng tỷ lệ thắng trượt (mục 6.5) |
| 16 | 7.4 | Gọi là "ẩn danh hóa"; nhầm salt với khóa bí mật; viện dẫn pháp lý cũ | Giả danh hóa, pepper, cập nhật Luật BVDLCN 2025 (mục 3.6) |
| 17 | 5.3 | Không nêu GRU và opset không được Unity Sentis hỗ trợ | Nêu ràng buộc và phương án xử lý (mục 8.1) |
| 18 | 9.3 | Ma trận độ phủ sai số mục, tự nhận "100%" | Lập lại ma trận có trạng thái và danh sách việc cần xác minh (Phần 10) |

---

## PHẦN 1: NỀN TẢNG VỀ GAME AI VÀ ĐIỀU CHỈNH ĐỘ KHÓ ĐỘNG (DDA)

### 1.1. Dynamic Difficulty Adjustment (DDA)

* **Định nghĩa.** DDA là cơ chế tự động thay đổi tham số của trò chơi (hành vi và chỉ số kẻ địch, tài nguyên, mức tha thứ sai sót) theo thời gian thực để thử thách phù hợp với năng lực hiện tại của người chơi.
* **Phân loại các hướng tiếp cận** (cần có trong Chương 1 để định vị đề tài):
  1. *Dựa trên luật (rule-based):* ngưỡng cứng trên chỉ số hiệu suất (ví dụ chết ≥ 2 lần thì giảm độ khó). Đơn giản, dễ giải thích, nhưng dễ dao động và không khái quát hóa.
  2. *Dựa trên mô hình hiệu suất/xác suất:* ước lượng khả năng thành công hoặc kỹ năng rồi chọn độ khó (họ mô hình Elo, Item Response Theory, mạng nơ-ron). **DDA-models thuộc nhóm này.**
  3. *Dựa trên học tăng cường (RL):* tối ưu chính sách điều chỉnh qua tương tác; cần mô phỏng hoặc log có propensity (xem Phần 9).
  4. *Dựa trên trạng thái cảm xúc/sinh lý (affect-based):* dùng tín hiệu sinh lý hoặc biểu cảm; nằm ngoài phạm vi telemetry của dự án.
* **Hệ thống tham chiếu thường được viện dẫn:** Hamlet (Hunicke), cơ chế AI Director của Left 4 Dead, thang độ khó ẩn của Resident Evil 4. Cần đọc nguồn gốc trước khi trích dẫn, vì nhiều hệ thống này là điều chỉnh nhịp độ/độ khó theo heuristic chứ không phải mô hình xác suất.
* **Điểm khác biệt của hệ thống:** thay vì phân loại người chơi thành nhãn tĩnh (`Beginner/Intermediate/Expert`) hoặc chỉ dự báo thắng/thua chung chung, hệ thống ước lượng **xác suất thành công của phân đoạn kế tiếp có điều kiện theo độ khó ứng viên**, rồi so sánh nhiều ứng viên trước khi hành động.

### 1.2. Phát biểu bài toán: dự báo can thiệp có điều kiện

**Đối tượng ước lượng.** Với mỗi điểm quyết định $t_k$, hệ thống ước lượng

$$\mu(h, c, d) \;=\; \Pr\big(Y_k = 1 \mid \mathcal{H}_{t_k}=h,\; c_{t_k}=c,\; \operatorname{do}(d_k = d)\big),$$

tức xác suất thành công *nếu* áp dụng độ khó $d$ cho phân đoạn $k$, và xấp xỉ bằng $\sigma\!\big(f_\theta(h,c,d)\big)$. Trong tài liệu kiến trúc, đại lượng này viết là $P(\text{segment\_success}=1 \mid \mathcal{H}_{60s}, \mathcal{C}_t, d_{\text{candidate}})$.

**Điều kiện nhận dạng.** Đại lượng trên là một *ước lượng can thiệp* chỉ khi mạng học đúng quan hệ nhân quả "đổi $d$ thì xác suất thắng đổi thế nào", không chỉ là tương quan quan sát. Gọi $Y_k(d)$ là kết quả tiềm năng (potential outcome) dưới độ khó $d$. Cần ba giả thiết:

* **(A1) Nhất quán (consistency):** kết quả quan sát bằng kết quả tiềm năng ứng với độ khó thực tế được áp dụng: $Y_k = Y_k(D_k)$.
* **(A2) Hoán đổi được có điều kiện (conditional ignorability):** $Y_k(d) \perp D_k \mid (\mathcal{H}, c)$ với mọi $d$. Đúng khi độ khó ở dữ liệu huấn luyện được **gán ngẫu nhiên**, hoặc được gán bởi chính sách chỉ phụ thuộc vào $(\mathcal{H}, c)$ và không có yếu tố ẩn (kỹ năng thật, trạng thái tâm lý) cùng ảnh hưởng đến $D_k$ và $Y_k$.
* **(A3) Dương tính (positivity/overlap):** $\Pr(D_k = d \mid \mathcal{H}, c) > 0$ với mọi ứng viên $d$ mà bộ điều khiển sẽ truy vấn.

Khi (A1)–(A3) đúng thì $\mu(h,c,d) = \mathbb{E}[Y \mid h, c, D=d]$, và học có giám sát trên dữ liệu quan sát cho ước lượng đúng.

**Hệ quả thiết kế dữ liệu** (đây là điều kiện để mô hình "đúng về mặt lý thuyết"):

1. Dữ liệu huấn luyện phải có **biến thiên độ khó được ngẫu nhiên hóa** tại các ranh giới phân đoạn. Nếu mọi hàng có $\delta = 0$ (phiên chơi cố định một profile), các thành phần delta của $d$ không đổi, mạng không thể học tác động của $d$. **[Khuyến nghị]**
2. Dữ liệu do bộ điều khiển tất định (DDA đang chạy) sinh ra vi phạm (A3) vì mỗi trạng thái chỉ gặp một độ khó, và dễ vi phạm (A2) do kỹ năng ẩn. Đây là lý do hệ thống chỉ dùng dữ liệu `fixed_randomized_same_domain` cho `empirical_prior` (mục 3.3).
3. Cần **log xác suất chọn hành động (propensity)** ngay từ giai đoạn thu thập, vì Phần 9 (bandit, off-policy evaluation) bắt buộc phải có. **[Khuyến nghị]**
4. Mạng dạng "$d$ là một đầu vào" thuộc nhóm **S-learner** trong ước lượng hiệu ứng điều trị không đồng nhất (CATE). Rủi ro đặc trưng của S-learner là *điều chuẩn kéo mô hình bỏ qua biến điều trị*: các ứng viên cho cùng một $p$, bộ điều khiển mất khả năng phân biệt. Do đó phải có **kiểm tra độ nhạy theo $d$** (mục 4.4).

Mô tả "phản thực" (counterfactual) chỉ nên dùng theo nghĩa thông thường "kịch bản giả định"; theo nghĩa nhân quả chặt chẽ (Pearl), điều này chỉ đúng khi các giả thiết trên được thỏa.

### 1.3. Lý thuyết Dòng chảy (Flow) và mục tiêu xác suất 0.70

* **Định nghĩa (Csikszentmihalyi).** Trạng thái Flow xảy ra khi mức thử thách cân bằng với kỹ năng, kèm mục tiêu rõ ràng và phản hồi tức thời. Thử thách $\gg$ kỹ năng gây lo âu/bực bội; thử thách $\ll$ kỹ năng gây nhàm chán.
* **Lượng hóa trong hệ thống.** Hệ thống chọn $P_{\text{target}} = 0.70$ (vùng mục tiêu $[0.60, 0.80]$ khi đánh giá).
* **Giải thích bằng mô hình Rasch (Item Response Theory).** Nếu $\Pr(\text{thắng}) = \sigma(\theta - b)$ với $\theta$ là năng lực và $b$ là độ khó, thì $P_{\text{target}} = 0.70 \iff \theta - b = \ln(7/3) \approx 0.847$ đơn vị logit; $P = 0.5$ ứng với $\theta = b$. Mô hình $f_\theta(\mathcal{H}, c, d)$ của hệ thống là một **tổng quát hóa phi tuyến của quan hệ năng lực – độ khó** này, và Elo/Glicko/TrueSkill là các họ cùng cấu trúc.
* **Lưu ý khoa học.** Con số 0.70 là **siêu tham số thiết kế**, không phải hệ quả suy ra từ lý thuyết Flow. Mục tiêu tối ưu phụ thuộc vào đích (học nhanh, thích thú, giữ chân người chơi). Ví dụ, "quy tắc 85%" (Wilson và cs., 2019) là kết quả cho tốc độ học của thuật toán học nhị phân, không phải cho mức độ thích thú khi chơi game. Báo cáo cần (i) nêu rõ đây là lựa chọn thiết kế, (ii) phân tích độ nhạy với $P_{\text{target}} \in \{0.6, 0.7, 0.8\}$, (iii) kiểm chứng bằng thang đo trải nghiệm (mục 7.3).
* **Định nghĩa `P_struggle`** xem mục 5.4. Ngưỡng $P_{\text{struggle}} \ge 0.75$ là ngưỡng trên *xác suất thất bại*, tương ứng $p(d_{\text{cur}}) \lesssim 0.25$, nằm xa đáng kể so với mục tiêu thất bại $0.30$; đây là vùng chết của bộ điều khiển (mục 6.4).

### 1.4. Game Telemetry, Segment Contract và định nghĩa kết quả

* **Event Log:** ghi *điều gì vừa xảy ra* tại thời điểm chính xác (`enemy_kill`, `player_hit`, `tower_build`, `death`).
* **Snapshot Log:** ghi *trạng thái tổng hợp* theo nhịp cố định ($\Delta = 5$ s), tạo chuỗi thời gian đều cho mạng hồi quy.
* **Segment Contract (`segments.py`)** **[Đã triển khai v2]**:
  * `EVENT_BOUNDED`: ranh giới gắn với cấu trúc tự nhiên của game (wave, room, checkpoint), thường 30–60 s. Khoảng nhãn là $[t, t_{\text{finish}}]$.
  * `FIXED_HORIZON`: cắt theo cửa sổ thời gian cố định (ví dụ 30 s sinh tồn) cho game thế giới mở/liên tục. Khoảng nhãn là $[t, t + h]$.
  * *Crash-Safe Guard* (`allow_horizon_fallback`): khi phiên đứt gãy trước sự kiện kết thúc, quay về ranh giới $[t, t+h]$ thay vì lỗi.
* **Định nghĩa nhãn và các trường hợp đặc biệt.** `segment_success` $= 1$ nếu người chơi hoàn thành phân đoạn theo tiêu chí của game, $= 0$ nếu thất bại. Cần định nghĩa rõ cho ba trường hợp mà tài liệu kiến trúc chưa chốt: **[Cần xác minh / Khuyến nghị]**
  1. **Quit giữa phân đoạn.** Đây là *kiểm duyệt có thông tin* (informative censoring): người chơi hay thoát sau đoạn quá khó, nên bỏ các mẫu này gây thiên lệch sống sót (survivorship bias). Khuyến nghị: ghi kết quả 3 lớp (`won/lost/quit`), huấn luyện trên `won/lost`, và chạy phân tích độ nhạy với hai quy tắc gán nhãn quit (coi là thua, hoặc loại bỏ). Với `Calibrator`, quy tắc gán nhãn phải đồng nhất với lúc huấn luyện.
  2. **Crash-fallback.** Nhãn theo horizon cố định có ngữ nghĩa khác nhãn theo sự kiện. Các cửa sổ dùng fallback cần được gắn cờ và loại khỏi tập kiểm thử.
  3. **Phân đoạn dài bất thường.** Nên ghi thời lượng phân đoạn để phân tích phụ.

### 1.5. Phạm vi cam kết và khung khái quát hóa ba bậc

* **Cam kết:** một *Universal Runtime Contract* cho phép tái sử dụng mô hình và pipeline điều khiển giữa các map, các bản cân bằng (patch), và chịu được khuyết thiếu nhóm tín hiệu.
* **Không cam kết:** không tuyên bố một mô hình zero-shot chạy cho mọi thể loại game không cần adapter. Mỗi game mới phải có **Semantic Adapter** theo hợp đồng và kiểm thử khả năng chơi.
* **Ba bậc khái quát hóa:**

| Bậc | Phạm vi | Giao thức đánh giá | Mức tuyên bố |
|:---:|---|---|---|
| 1 | Cùng game, **người chơi/phiên mới** | Group split theo `player_id_hash` | Tuyên bố chính |
| 2 | **Map/patch mới**, cùng game | Zero-shot, rồi few-shot calibration (Leave-Session-Out) | Tuyên bố chính, có điều kiện |
| 3 | **Thể loại khác** qua adapter ngữ nghĩa | Mini-game khác, đánh giá thăm dò | Giả thuyết/khám phá, không khẳng định |

* **Zero-Shot vs Few-Shot.** *Zero-shot* nghĩa là không sửa trọng số **và không dùng dữ liệu miền đích để fit bất kỳ tham số nào** (kể cả `OnlineTemperatureCalibrator`). *Few-shot adaptation* là dùng một lượng nhỏ dữ liệu miền đích (20–60 segment) để hiệu chỉnh tầng nhẹ; phải đánh giá bằng Leave-Session-Out, không bao giờ dùng dữ liệu của chính phiên kiểm thử để fit rồi báo cáo như zero-shot.

---

## PHẦN 2: BIỂU DIỄN ĐẶC TRƯNG CANONICAL, DỮ LIỆU THIẾU VÀ CHUẨN HÓA

### 2.1. Hợp đồng 21 tín hiệu ngữ nghĩa canonical **[Đã triển khai v2]**

Hợp đồng `canonical_feature_contract.json` định nghĩa $F = 21$ tín hiệu, chia thành **4 nhóm nghiệp vụ với kích thước 4 / 4 / 4 / 9**:

| Nhóm | Số biến | Tín hiệu | Bounded $[0,1]$ |
|---|:---:|---|---|
| `progress_tempo` | 4 | `progress_ratio`, `progress_velocity`, `time_budget_ratio`, `tempo_rate` | `progress_ratio` |
| `vitality_error` | 4 | `vitality_ratio`, `vitality_loss_rate`, `critical_error_rate`, `failure_cost_ratio` | `vitality_ratio` |
| `capability_economy` | 4 | `resource_reserve_ratio`, `resource_utilization_rate`, `capability_ratio`, `threat_pressure_ratio` | (không) |
| `human_cognitive_motor` | 9 | `action_frequency`, `invalid_action_rate`, `idle_ratio`, `reaction_latency`, `error_slope`, `recent_segment_win_rate`, `recent_average_difficulty`, `difficulty_adjusted_win_rate`, `efficiency_ratio` | `invalid_action_rate`, `idle_ratio`, `recent_segment_win_rate` |

Có đúng **5 biến bounded** (`progress_ratio`, `vitality_ratio`, `invalid_action_rate`, `idle_ratio`, `recent_segment_win_rate`); 16 biến còn lại là *unbounded* và đi qua chuẩn hóa co cụm (mục 2.6). `time_budget_ratio` được xếp vào unbounded vì có thể vượt 1.0 khi bù giờ; `difficulty_adjusted_win_rate` có miền $[-1,1]$ nên chuẩn hóa shrinkage để không bị clip mất phần âm.

**Phân tích nhóm `human_cognitive_motor` (điểm cần nêu chính xác trong báo cáo).** Nhóm này thực chất gồm ba loại biến khác bản chất:

* **(a) 5 tín hiệu thao tác thuần:** `action_frequency`, `invalid_action_rate`, `idle_ratio`, `reaction_latency`, `error_slope`.
* **(b) 3 tín hiệu lịch sử/hiệu suất:** `recent_segment_win_rate`, `recent_average_difficulty`, `difficulty_adjusted_win_rate`. Ngữ nghĩa phụ thuộc vào thang độ khó của từng game; chúng khả dụng muộn (mask $=0$ ở đầu trận).
* **(c) 1 tín hiệu hiệu quả:** `efficiency_ratio` (đầu ra trên chi phí thao tác), cần định nghĩa riêng cho mỗi game trong adapter.

Lập luận "tín hiệu con người có quy luật tương đồng giữa các game (phản ứng hoảng loạn, do dự, làm chủ)" **chỉ áp dụng cho nhóm (a)**. Đây vẫn là **giả thuyết** (độ lớn của `action_frequency` phụ thuộc thể loại, thiết bị nhập), được kiểm chứng ở bậc 3 và giảm thiểu một phần bằng chuẩn hóa theo phiên. Ngoài ra vì độ phủ có trọng số nhóm đều $25\%$ mà nhóm cuối có 9 biến, mỗi biến của nhóm này chỉ có trọng số $0.25/9 \approx 2.8\%$ so với $6.25\%$ ở các nhóm còn lại (mục 5.3).

### 2.2. Biểu diễn bất biến theo miền: đúng nghĩa và giới hạn

* **Hai nghĩa của "domain-invariant".** (i) *Căn chỉnh lược đồ thủ công (schema alignment):* ánh xạ biến đặc thù của từng game (`gold`, `tower_count`, `wave`) sang từ vựng ngữ nghĩa chung; đây là điều hệ thống làm. (ii) *Biểu diễn bất biến được học:* huấn luyện bộ mã hóa sao cho phân phối biểu diễn giống nhau giữa các miền (DANN, IRM); hệ thống **không** làm điều này. Báo cáo nên gọi hệ thống là *canonical feature space (không gian đặc trưng chuẩn hóa ngữ nghĩa)* để tránh nhầm.
* **Giới hạn lý thuyết.** Theo cận của Ben-David và cs., sai số miền đích bị chặn bởi sai số miền nguồn cộng một nửa khoảng cách phân phối giữa hai miền và một số hạng $\lambda$ (sai số tối ưu chung). Căn chỉnh lược đồ có thể giảm số hạng khoảng cách, nhưng **không bảo đảm $\lambda$ nhỏ**; khi quan hệ $P(Y\mid X)$ đổi (concept drift) thì $\lambda$ lớn và căn chỉnh thuần ngữ nghĩa không đủ.
* **Hai biến ngữ cảnh chưa thật sự bất biến theo miền:** **[Khuyến nghị]**
  * `current_difficulty` là chỉ số profile; "profile 3" ở map này không tương đương map khác. Dạng delta trong $d$ bất biến, nhưng giá trị tuyệt đối thì không.
  * `normalized_segment_index` $= \text{clip}\big(\text{segment\_index}/\text{total\_segments},0,1\big)$ cần biết `total_segments` trước; với game vô hạn/sinh tồn thì không xác định, nên cần mask hoặc giá trị thay thế.

### 2.3. Dữ liệu thiếu: cơ chế, mặt nạ và nguyên tắc không bịa dữ liệu

* **Ba dạng thiếu trong hệ thống:** (1) *cấu trúc:* game không có tín hiệu đó; (2) *chưa khả dụng:* lịch sử thắng ở đầu trận; (3) *ngẫu nhiên/hỏng:* mất snapshot, lỗi mạng.
* **Cơ chế thiếu (Little & Rubin).** Gọi $R$ là chỉ báo quan sát. *MCAR:* $\Pr(R \mid X) = \Pr(R)$. *MAR:* $\Pr(R \mid X)$ chỉ phụ thuộc phần quan sát được của $X$. *MNAR:* phụ thuộc chính giá trị bị thiếu. Với dạng (1) và (2), mặt nạ là hàm tất định của game/thời điểm, nên đưa mặt nạ vào mô hình là hợp lý và có thông tin. Với dạng (3), cần kiểm tra thiếu có liên quan đến tình huống khó hay không (MNAR có thể làm lệch).
* **Biểu diễn `(value, mask)`.** Hàm `canonicalize` gán: đặc trưng hữu hạn $\Rightarrow$ `value = x, mask = 1`; vắng mặt/NaN/`available=False` $\Rightarrow$ `value = 0, mask = 0`. Cặp $(0, 1)$ (giá trị quan sát bằng 0, ví dụ máu bằng 0) khác hoàn toàn $(0, 0)$ (thiếu).
* **Nguyên tắc không nội suy (Non-Fabrication).** Không dùng nội suy tuyến tính để chẻ dữ liệu thưa thành chuỗi 5 s, vì (i) tạo trạng thái chưa từng xảy ra (giả định mất máu đều trong khi thực tế mất máu ở giây 28), và (ii) nội suy giữa hai mốc dùng thông tin *sau* thời điểm đang xét, tức rò rỉ tương lai. Giá trị không có giữ nguyên `NaN` $\Rightarrow$ `mask = 0`.
* **Bộ lọc hai tầng của `build_window_rows`** **[Đã triển khai v2]**: (1) thiếu theo *chiều đặc trưng* giữ nguyên (mask $=0$), nhưng cửa sổ cần ít nhất 5 đặc trưng khả dụng; (2) thiếu theo *trục thời gian* chỉ tính trên tập đặc trưng có ít nhất một quan sát trong cửa sổ, và loại cửa sổ nếu tỷ lệ NaN trung bình $> 35\%$. Ngưỡng 35% và 5 đặc trưng là **siêu tham số cần phân tích độ nhạy**.
* **Điều kiện để mô hình thật sự "chịu được khuyết nhóm" (rất quan trọng).** Vì token bị mask nhận trọng số pooling bằng 0 (mục 4.3), khi mất cả một nhóm thì $z_t$ là trung bình của các token *còn lại*, một phân phối đầu vào mà GRU chưa từng thấy nếu huấn luyện luôn đủ nhóm. Do đó cần **feature-group dropout trong huấn luyện** (mỗi bước, mỗi nhóm bị che với xác suất nhỏ, luôn giữ ít nhất 2 nhóm). Hàm `masked_feature_dropout` hiện được mô tả là công cụ mô phỏng cho Benchmark B; cần xác minh nó có được dùng trong huấn luyện không. **[Cần xác minh / Khuyến nghị]**
* **Lấy mẫu không đều.** GRU giả định bước thời gian đều. Khi $\Delta t$ dao động, các đặc trưng tốc độ đã chia cho $\Delta t$ thực; mở rộng có thể đưa $\Delta t$ làm đặc trưng hoặc dùng GRU-D. **[Mở rộng]**

### 2.4. Đặc trưng động học (sai phân hữu hạn) và làm mượt

Các đặc trưng phái sinh (`add_derived_features`) dùng khoảng cách thời gian thực $\Delta t$ giữa hai snapshot liên tiếp **[Đã triển khai v2]**:

$$v_{\text{progress}} = \frac{\Delta\,\text{progress\_ratio}}{\Delta t},\quad r_{\text{hp\_loss}} = \frac{\max(0,-\Delta\,\text{hp\_ratio})}{\Delta t},\quad r_{\text{gold\_spend}} = \frac{\max(0,-\Delta\,\text{gold})}{\Delta t},$$

$$r_{\text{tower\_build}} = \frac{\max(0,\Delta\,\text{tower\_count})}{\Delta t},\qquad \text{error\_slope} = \frac{\Delta\,\text{enemy\_leaks\_30s}}{\Delta t}.$$

* **Chỉnh lưu $\max(0,\cdot)$** biến sai phân thành *tốc độ mất/tiêu/xây* không âm; hồi máu hoặc nhận vàng không bị tính là âm.
* **Khuếch đại nhiễu.** Sai phân tính từ hai điểm khuếch đại nhiễu tần số cao (phương sai nhân với $\approx 2/\Delta t^2$ nếu nhiễu độc lập), nên các biến này đi qua chuẩn hóa và clip $[-8, 8]$.
* **Đơn vị.** `enemy_leaks_30s` là đếm trong cửa sổ trượt 30 s, nên `error_slope` ở đây là tốc độ thay đổi của một *đếm trượt*, không phải đạo hàm bậc hai của số lỗi tích lũy. Hợp đồng cần ghi rõ đơn vị. **[Khuyến nghị]**
* **EMA** (nếu dùng để làm mượt): $\text{EMA}_t = \alpha x_t + (1-\alpha)\text{EMA}_{t-1}$. Với lấy mẫu không đều, đặt $\alpha_t = 1 - e^{-\Delta t/\tau_{\text{ema}}}$; khởi tạo $\text{EMA}_0 = x_0$; phải là làm mượt **nhân quả** (chỉ dùng quá khứ), không dùng cửa sổ trung tâm vì rò rỉ tương lai. **[Khuyến nghị nếu dùng]**

### 2.5. Biểu diễn độ khó liên tục: log-ratio chuẩn hóa

Với mỗi tham số độ khó $j$ (tốc độ quái, máu quái, ...), độ biến thiên ứng viên so với hiện tại:

$$\delta_j = \operatorname{clip}\!\left(\frac{s_j \cdot \ln\!\big(a_j^{\text{cand}}/a_j^{\text{cur}}\big)}{\ln r_{\max}},\; -1,\; 1\right),\qquad s_j \in \{+1,-1\}.$$

**Tính chất (đều chứng minh được trực tiếp từ $\ln$):**

1. **Phản đối xứng:** $\delta(a,b) = -\delta(b,a)$, vì $\ln(b/a) = -\ln(a/b)$. Ví dụ $\ln 1.25 = 0.2231 = -\ln 0.80$.
2. **Bất biến theo đơn vị đo:** $\delta(\lambda a, \lambda b) = \delta(a,b)$ với mọi $\lambda>0$. HP $100\to120$ và $10\,000\to12\,000$ cho cùng $\delta$.
3. **Cộng tính trước khi clip:** $\ln(c/a) = \ln(b/a) + \ln(c/b)$, tức nhiều bước nhân liên tiếp ứng với tổng $\delta$. Phép clip phá cộng tính ở biên.
4. **Chặn biên:** $\delta \in [-1,1]$; $\delta = \pm 1$ tương ứng tỉ lệ $r_{\max}^{\pm 1}$.

**Điều kiện áp dụng cần ghi vào hợp đồng.** **[Khuyến nghị]**

* Tham số phải dương ($\ln$ không xác định với $0$). Với tham số cộng tính hoặc có thể bằng 0 (số quái xuất hiện thêm), dùng sai phân chuẩn hóa tuyến tính thay thế.
* **Quy ước dấu $s_j$:** nên chọn $s_j$ sao cho $\delta > 0$ luôn nghĩa "khó hơn" (tăng máu quái: $s = +1$; tăng tài nguyên hỗ trợ: $s = -1$). Nếu không, không thể phát biểu tính đơn điệu theo từng tọa độ (mục 4.4).

**Bố cục vector 10 chiều** **[Đã triển khai v2]**: các chỉ số 0–4 là giá trị, 5–9 là mặt nạ.

| Chỉ số | Biến | Miền | Ghi chú |
|:---:|---|---|---|
| 0 | `relative_step` | $[-1,1]$ | Bước nhảy profile tương đối; với ứng viên $\{c-1,c,c+1\}$ nhận $\{-1,0,1\}$ |
| 1 | `pressure_delta` | $[-1,1]$ | Log-ratio áp lực kẻ địch |
| 2 | `forgiveness_delta` | $[-1,1]$ | Log-ratio mức tha thứ sai sót |
| 3 | `resource_delta` | $[-1,1]$ | Log-ratio tiếp tế/hồi phục |
| 4 | `empirical_prior` | $[0,1]$ | Xác suất thắng tiên nghiệm của profile ứng viên |
| 5–8 | mask của 0–3 | $\{1\}$ | Luôn bằng 1 |
| 9 | mask của `empirical_prior` | $\{0,1\}$ | $=1$ chỉ khi cùng domain/patch; $=0$ khi zero-shot |

**`empirical_prior` là một dạng *target encoding*, nên có nguy cơ rò rỉ nhãn.** Nó là thống kê của chính nhãn ($\pi_0(d) \approx \Pr(Y=1\mid\text{profile }d)$) đưa vào làm đầu vào. Ba yêu cầu: **[Khuyến nghị]**

1. Chỉ tính từ **người chơi thuộc tập train** (hoặc out-of-fold khi dùng cross-validation), không bao giờ dùng người chơi của validation/test.
2. Làm mượt theo Beta–Binomial ($\hat\pi_0 = (\text{wins}+a)/(\text{n}+a+b)$) khi mẫu ít.
3. Trong huấn luyện phải có các hàng với mask $=0$ (che ngẫu nhiên) thì đường zero-shot mới được học; nếu không, mô hình chưa từng thấy `mask[9] = 0`. Đồng thời kiểm tra mô hình không "đi tắt" bằng chỉ dựa vào $\pi_0$ (ablation: bỏ chỉ số 4 và 9).

### 2.6. `CausalSessionNormalizer`: Welford, co cụm Bayes và phương sai hỗn hợp

**Quy trình cho từng tín hiệu $j$ tại mỗi snapshot** **[Đã triển khai v2]**:

* **Bounded** (5 biến): $\tilde x = \text{clip}(x, 0, 1)$, giữ nguyên ngữ nghĩa vật lý. Nên đếm và ghi log số lần clip thực sự xảy ra (giá trị ngoài $[0,1]$ thường là dữ liệu lỗi, nên đưa vào vùng cách ly thay vì clip âm thầm).
* **Unbounded** (16 biến), khi `mask == 1`:
  1. Cập nhật Welford; 2. Tính trọng số co cụm; 3. Tính trung bình và phương sai hỗn hợp; 4. Áp sàn độ lệch chuẩn; 5. Chuẩn hóa và clip:

$$w = \frac{n}{\kappa + n},\qquad \mu = (1-w)\mu_{g} + w\mu_{s},$$
$$\sigma^2 = (1-w)\sigma_{g}^2 + w\sigma_{s}^2 + w(1-w)(\mu_{g}-\mu_{s})^2,$$
$$\sigma_{\text{eff}} = \max\!\big(\sqrt{\sigma^2},\,\sigma_{\min}\big),\qquad \tilde x = \operatorname{clip}\!\left(\frac{x-\mu}{\sigma_{\text{eff}}},\,-8,\,8\right).$$

Khi `mask == 0`: $\tilde x = 0$, mask $=0$, và **không cập nhật** thống kê. $\kappa$ là `prior_weight`, $\sigma_{\min}=0.05$. *(Giá trị $\kappa$ [Cần xác minh].)*

**Mệnh đề 1 (Welford).** Với $n \leftarrow n+1$, $\delta = x - \mu$, $\mu \leftarrow \mu + \delta/n$, $M_2 \leftarrow M_2 + \delta\,(x-\mu_{\text{mới}})$, thì $\mu$ là trung bình mẫu và $M_2/n$ là phương sai tổng thể (population) của $n$ quan sát. Bộ nhớ $O(1)$, và tránh hủy số học (catastrophic cancellation) của công thức $\mathbb{E}[X^2]-(\mathbb{E}X)^2$. Cần quy ước rõ chia $n$ hay $n-1$; với $n=1$ đặt $\sigma_s^2 = 0$ (không xác định theo $n-1$). Nên kiểm tra bằng đối chiếu số (đã kiểm: trên 1000 mẫu, kết quả trùng `numpy` đến sai số máy).

**Mệnh đề 2 (phương sai hỗn hợp).** Nếu coi $X$ lấy từ phân phối "game" với xác suất $1-w$ và phân phối "phiên" với xác suất $w$, thì $\operatorname{Var}(X) = \sum_i p_i(\sigma_i^2 + \mu_i^2) - \mu^2$. Khai triển:
$$(1-w)\mu_g^2 + w\mu_s^2 - \big((1-w)\mu_g + w\mu_s\big)^2 = w(1-w)(\mu_g - \mu_s)^2,$$
nên công thức ở trên đúng. *(Đã kiểm bằng mô phỏng Monte Carlo 2 triệu mẫu: 1.614 so với 1.615 theo công thức.)* Số hạng chéo lớn nhất khi $w = 0.5$ và bằng $\tfrac14(\mu_g-\mu_s)^2$.

**Diễn giải Bayes và giới hạn.** Trọng số $w = n/(\kappa+n)$ chính là hậu nghiệm của mô hình chuẩn–chuẩn liên hợp với phương sai đã biết khi coi giá trị tiên nghiệm có trọng số $\kappa$ quan sát giả (pseudo-count). Tuy nhiên công thức *phương sai* là **moment-matching của hỗn hợp**, không phải phương sai hậu nghiệm của mô hình đó. Cả hai đều là heuristic hợp lý cho thang chuẩn hóa; báo cáo nên gọi đúng tên "shrinkage kiểu empirical-Bayes + phương sai hỗn hợp".

**Ba điểm lý thuyết cần xử lý hoặc nêu rõ:**

1. **Chuẩn hóa theo phiên làm mất thông tin mức tuyệt đối.** Khi $n \to \infty$, $w \to 1$ và $\tilde x \to (x-\mu_s)/\sigma_s$: đặc trưng chỉ còn là độ lệch *so với chính người chơi đó*. Người chơi liên tục vật lộn có `vitality_loss_rate` cao ổn định sẽ cho $\tilde x \approx 0$. Đây là cái giá của việc loại ảnh hưởng của thang đo giữa map/patch. Mức tuyệt đối vẫn còn qua 5 biến bounded, `recent_segment_win_rate`, `current_difficulty`. Cần **ablation 3 biến thể** chuẩn hóa: (i) chỉ dùng prior game, (ii) chỉ dùng thống kê phiên, (iii) co cụm (hệ thống hiện tại). **[Khuyến nghị]**
2. **Sàn $\sigma_{\min}=0.05$ không nhất quán về thứ nguyên.** Hằng số tuyệt đối chỉ hợp lý với biến có thang cỡ đơn vị. Nếu thang thật của một tốc độ là $\sigma_g = 0.005$, sàn làm co $\tilde x$ đi 10 lần (thay đổi 1σ thật chỉ còn $0.1$). Đề xuất sàn theo từng đặc trưng:
$$\sigma_{\text{eff},j} = \max\!\big(\sqrt{\sigma_j^2},\;\rho\,\sigma_{g,j},\;\epsilon_j\big),\quad \rho \approx 0.25.$$
Sàn vẫn cần (chống chia cho 0), nhưng phải theo thang của từng biến; clip $\pm 8$ mới là cơ chế chống ngoại lai. **[Khuyến nghị]**
3. **Tương đương train–serve (train–serve parity).** Mỗi snapshot phải được chuẩn hóa bằng thống kê **tính đến chính snapshot đó** (theo thứ tự thời gian của phiên), cả lúc huấn luyện và lúc chạy; không được tính lại thống kê tại cuối cửa sổ hoặc dùng cả phiên. Cần quy ước nhất quán về việc quan sát hiện tại có được đưa vào thống kê trước khi chuẩn hóa. Test đơn vị: chạy cùng một chuỗi qua đường train và đường inference, so sánh từng phần tử. **[Cần xác minh]**

**Định nghĩa `difficulty_adjusted_win_rate`.** Hiệu số giữa tỷ lệ thắng gần đây và kỳ vọng thắng tiên nghiệm của các độ khó đã trải qua: $\text{DAWR} = \bar y_{\text{recent}} - \overline{\pi_0(d_{\text{recent}})} \in [-1,1]$ (*quan sát trừ kỳ vọng*, cùng tinh thần với điểm Elo). Mask $=0$ nếu không có $\pi_0$. Công thức chính xác **[Cần xác minh]**.

---

## PHẦN 3: TOÀN VẸN DỮ LIỆU, THỐNG KÊ VÀ DỊCH CHUYỂN PHÂN PHỐI

### 3.1. Rò rỉ dữ liệu (Data Leakage) và cô lập ranh giới nhân quả

**Định nghĩa hình thức.** Gọi $\mathcal{F}_t$ là tập thông tin quan sát được đến hết thời điểm quyết định $t$. Một đặc trưng đầu vào hợp lệ phải là hàm chỉ của $\mathcal{F}_t$; nhãn là hàm của các sự kiện trong $[t, t_{\text{finish}}]$. Rò rỉ xảy ra khi có đường thông tin từ ngoài $\mathcal{F}_t$ (tương lai, hoặc tập validation/test) vào đặc trưng, nhãn, tham số tiền xử lý hay lựa chọn mô hình.

**Các dạng rò rỉ và biện pháp chặn.**

| Dạng | Ví dụ | Biện pháp | Trạng thái |
|---|---|---|---|
| Rò rỉ nhãn | Gán `Won/Lost` của cả trận cho mọi snapshot | Mỗi cửa sổ chỉ nhận nhãn của phân đoạn ngay sau $t$ | [Đã triển khai v2] |
| Rò rỉ đặc trưng tương lai | `deaths_in_horizon`, `damage_taken_horizon`, `segment_success`, `match_result` làm đầu vào | Danh mục cấm đặc trưng (Banned Features) | [Đã triển khai v2] |
| Rò rỉ chuẩn hóa | Fit scaler bằng trung bình/phương sai cả ván hoặc cả tập test | `CausalSessionNormalizer` chỉ dùng snapshot đến $t$ | [Đã triển khai v2] |
| Rò rỉ do nội suy | Nội suy giữa mốc quá khứ và mốc tương lai | Nguyên tắc không nội suy (mục 2.3) | [Đã triển khai v2] |
| Rò rỉ do chia tập | Hai cửa sổ chồng lấn 75% rơi vào train và test | Group split theo người chơi | [Đã triển khai v2] |
| Rò rỉ do target encoding | `empirical_prior` tính cả từ người chơi val/test | Chỉ tính trên train, out-of-fold (mục 2.5) | [Khuyến nghị] |
| Rò rỉ do chọn mô hình | Fit nhiệt độ hoặc dừng sớm bằng tập test | Dùng validation cho mọi quyết định; test chỉ báo cáo một lần | [Khuyến nghị] |

### 3.2. Chia tập theo nhóm (Group Split) và tự tương quan thời gian

* **Mức chồng lấn.** Với cửa sổ $L$ snapshot và bước nhảy $s$, hai mẫu liên tiếp chia sẻ $(L-s)/L$ dữ liệu: $s=1$ cho $11/12 = 91.7\%$; $s=3$ (15 s, như `build_window_rows`) cho $9/12 = 75\%$; chỉ $s = L = 12$ mới không chồng lấn.
* **Hệ quả: dữ liệu không i.i.d.** Các mẫu trong một phiên và một người chơi có tương quan mạnh. Chia ngẫu nhiên theo hàng khiến mô hình thuộc lòng dấu vết người chơi/đoạn gameplay kề nhau, và đo lường quá lạc quan. Giải pháp: chia **tách biệt theo người chơi** (`player_id_hash`, fallback `session_id`) thành train/val/test $70/15/15$ **[Đã triển khai v2]**.
* **Cần lưu ý thêm:**
  1. Fallback `session_id` yếu hơn `player_id`: cùng một người có thể có nhiều phiên và rò sang test. Hạn chế dùng; nếu buộc phải dùng, nêu rõ trong giới hạn.
  2. `stride = 3` chỉ giảm dư thừa, **không** khiến mẫu độc lập. Biện pháp bảo đảm thật sự là group split. Khi báo cáo khoảng tin cậy, dùng **cluster bootstrap theo người chơi** (resample *người chơi*, không resample hàng).
  3. Kích thước mẫu hiệu dụng khi tự tương quan: $n_{\text{eff}} = n/(1 + 2\sum_{k\ge 1}\rho_k)$ với $\rho_k$ là tự tương quan trễ $k$; thường nhỏ hơn $n$ nhiều.
  4. Nếu số người chơi ít, một lần chia 70/15/15 cho phương sai cao; dùng `StratifiedGroupKFold` lặp lại và báo cáo trung bình ± độ lệch giữa các fold. **[Khuyến nghị]**
  5. Với bậc 2 (patch mới), dùng **chia theo thời gian/patch** (train patch cũ, test patch mới), không chỉ chia theo người.

### 3.3. Gây nhiễu (confounding), positivity, ngẫu nhiên hóa và nghịch lý Simpson

**Cơ chế gây sai.** Gọi $S$ là kỹ năng (không quan sát trực tiếp). Nếu dữ liệu thu khi DDA hoạt động, người giỏi được đẩy lên độ khó cao: $S \to D$, và $S \to Y$, $D \to Y$. Khi đó $\Pr(Y=1\mid D=d) \neq \Pr(Y=1\mid \operatorname{do}(D=d))$ vì $S$ là biến gây nhiễu (backdoor path $D \leftarrow S \to Y$).

**Ví dụ số cho nghịch lý Simpson.**

| Nhóm kỹ năng | Profile 0 (dễ) | Profile 4 (khó) |
|---|---|---|
| Người mới | 40/80 thắng (50%) | 5/20 thắng (25%) |
| Người giỏi | 19/20 thắng (95%) | 64/80 thắng (80%) |
| **Gộp** | **59/100 (59%)** | **69/100 (69%)** |

Trong *từng* nhóm, profile 4 khó hơn (25% < 50%, 80% < 95%), nhưng số liệu gộp lại cho thấy profile 4 *dễ thắng hơn* (69% > 59%), vì người giỏi chiếm đa số ở profile 4. Mô hình học trên số liệu thô sẽ học sai quy luật "càng khó càng dễ thắng".

**Biện pháp.**

1. **Ngẫu nhiên hóa độ khó** (`fixed_randomized_same_domain`): loại bỏ cạnh $S \to D$ theo thiết kế, nên $\Pr(Y\mid D)$ trở lại bằng $\Pr(Y\mid \operatorname{do}(D))$. Chỉ dữ liệu này được dùng để tính `empirical_prior`; nếu chạy zero-shot sang map/patch mới thì `mask = 0` **[Đã triển khai v2]**.
2. **Positivity.** Với mọi trạng thái $(h,c)$ mà bộ điều khiển sẽ truy vấn, các ứng viên $\{c-1, c, c+1\}$ đều phải có xác suất được gán dương trong dữ liệu. Bộ điều khiển tất định vi phạm điều này, nên cần exploration có kiểm soát khi thu thập (ví dụ ngẫu nhiên hóa $\epsilon$ trong biên an toàn). **[Khuyến nghị]**
3. **Vòng phản hồi (feedback loop).** Khi triển khai, chính sách đổi phân phối dữ liệu tương lai. Huấn luyện lại trên dữ liệu có chính sách tạo ra phải kèm propensity (Phần 9). **[Khuyến nghị]**
4. **Điều kiện bổ sung của mô hình: mạng phải thật sự dùng $d$.** Xem kiểm tra độ nhạy theo $d$ (mục 4.4).

### 3.4. Dịch chuyển phân phối: covariate shift, label shift, concept drift

| Loại | Điều thay đổi | Điều giữ nguyên | Ví dụ |
|---|---|---|---|
| **Covariate shift** | $P(X)$ | $P(Y\mid X)$ | Patch tăng gấp đôi lượng vàng; map đông quái hơn |
| **Label (prior) shift** | $P(Y)$ | $P(X\mid Y)$ | Map mới dễ hơn nên tỷ lệ thắng nền tăng |
| **Concept drift** | $P(Y\mid X)$ | — | Cơ chế sát thương đổi, cùng lượng máu nhưng dễ chết hơn |

**Mỗi thành phần của hệ thống sửa được gì.**

* `CausalSessionNormalizer` giảm tác động của dịch chuyển về *vị trí và thang* của biên $P(X)$ (một dạng covariate shift), không sửa được thay đổi hình dạng phân phối hay concept drift.
* Hiệu chuẩn nhiệt độ sửa *độ tự tin* (thang của logit). **Nó không sửa được lệch (bias) và không sửa được thứ hạng** khi concept drift làm đổi quan hệ $X\to Y$; trường hợp này cần huấn luyện lại hoặc few-shot fine-tuning ngoài runtime.
* **Hiệu chỉnh label shift có công thức đóng** (Saerens và cs.): nếu $P(X\mid Y)$ không đổi thì odds hậu nghiệm nhân với tỷ số odds tiên nghiệm, tức
$$z' = z + \ln\frac{\pi_T}{1-\pi_T} - \ln\frac{\pi_S}{1-\pi_S},$$
trong đó $\pi_S, \pi_T$ là tỷ lệ thắng nền ở miền nguồn và đích. Đây chính là phép *dịch* mà nhiệt độ $\tau$ không thể thực hiện (mục 5.2). **[Khuyến nghị]**

**Phát hiện và giám sát.** PSI hoặc KS cho từng đặc trưng; MMD cho phân phối đa biến; ADWIN/DDM cho luồng sai số; giám sát ECE/Brier trượt trên cửa sổ gần nhất. **[Khuyến nghị]** *Importance weighting* $w(x) = p_T(x)/p_S(x)$ là kỹ thuật chuẩn cho covariate shift nhưng nằm ngoài phạm vi v2. **[Mở rộng]**

### 3.5. Căn chỉnh nhãn và điểm quyết định (decision-point alignment)

* **Quan sát.** `build_window_rows` trượt cửa sổ với `stride = 3` trên toàn phiên, nên nhiều cửa sổ kết thúc ở thời điểm *bất kỳ*, trong khi lúc triển khai bộ điều khiển chỉ chạy tại **safe checkpoint**. Phân phối $(h, c)$ lúc huấn luyện khác phân phối lúc triển khai (train–serve skew); với cửa sổ kết thúc giữa phân đoạn thì "phân đoạn kế tiếp" còn cách một khoảng thời gian chưa quan sát.
* **Khuyến nghị.** **[Khuyến nghị]**
  1. Huấn luyện có thể dùng mọi cửa sổ, nhưng **đánh giá chỉ trên các cửa sổ kết thúc tại điểm quyết định** (hoặc cách nó không quá một stride), vì đây là phân phối triển khai.
  2. Thêm đặc trưng ngữ cảnh `time_to_segment_start` hoặc `time_since_checkpoint` để mạng biết khoảng cách tới phân đoạn được dự báo.
  3. Ghi rõ số cửa sổ căn chỉnh theo điểm quyết định (thường chỉ khoảng một cửa sổ mỗi phân đoạn, nên dữ liệu ít) để diễn giải độ rộng khoảng tin cậy.

### 3.6. Quản trị dữ liệu, giả danh hóa và khung pháp lý

* **Schema Contract.** `canonical_feature_contract.json` khóa chặt thứ tự đặc trưng, kiểu, biên và chính sách thiếu giữa bộ thu thập (Unity logger) và pipeline huấn luyện (Python) **[Đã triển khai v2]**.
* **Vùng cách ly (Quarantine) và Audit Trail.** Log lỗi parse, sai phiên bản schema hoặc vi phạm miền vật lý (`hp > hpMax`, $\Delta t \le 0$) được chuyển sang `quarantine/` thay vì sửa âm thầm, để bảo đảm truy vết và tái lập; bản ghi quyết định/mô hình có dấu thời gian và tác nhân. **[Khuyến nghị quản trị dữ liệu; Cần xác minh việc đã có trong hạ tầng log]**
* **Tạo mã người chơi bằng `HMAC-SHA256`:**
$$\text{playerIdHash} = \text{HMAC\_SHA256}\big(K_{\text{secret}},\; \text{internal\_player\_id}\big).$$
  * Đây là **giả danh hóa (pseudonymization)**, không phải ẩn danh hóa: cùng một người luôn cho cùng một mã, nên vẫn liên kết được theo thời gian (điều kiện cần cho group split và phân tích học tập). Dữ liệu giả danh hóa vẫn được coi là dữ liệu cá nhân, vì chủ thể kiểm soát khóa có thể nối lại.
  * Phân biệt **salt** (ngẫu nhiên, không bí mật, theo từng bản ghi) với **khóa bí mật/pepper** $K_{\text{secret}}$ (bí mật, dùng chung). HMAC dùng khóa bí mật. Rủi ro chính của SHA-256 trần là *tấn công từ điển/brute-force* trên ID có entropy thấp; khóa bí mật loại bỏ khả năng đó nếu khóa không lộ.
  * Quản lý khóa: lưu tách khỏi dữ liệu; xoay khóa làm mất liên kết dọc, nên cần kế hoạch (ví dụ giữ khóa theo giai đoạn nghiên cứu).
  * Chính hành vi telemetry cũng có thể là "dấu vân tay"; cần tối thiểu hóa dữ liệu và thời hạn lưu trữ.
* **Khung pháp lý Việt Nam (kiểm tra tháng 11/2026; không phải tư vấn pháp lý).** Luật Bảo vệ dữ liệu cá nhân (Luật số 91/2025/QH15, thông qua ngày 26/6/2025) có hiệu lực từ 01/01/2026, và Nghị định 356/2025/NĐ-CP hướng dẫn thi hành thay thế Nghị định 13/2023/NĐ-CP từ cùng ngày. Báo cáo nên viện dẫn khung hiện hành, nêu căn cứ xử lý (sự đồng ý rõ ràng của người chơi cho mục đích nghiên cứu), nguyên tắc tối thiểu hóa, thời hạn lưu trữ, và xử lý người chơi dưới 18 tuổi. Nên đối chiếu văn bản chính thức và hỏi đơn vị pháp chế/hội đồng đạo đức của trường trước khi thu thập dữ liệu người chơi thật.

---

## PHẦN 4: HỌC MÁY VÀ KIẾN TRÚC HỌC SÂU

### 4.1. Các mô hình nền (baseline)

* **`MajorityModel` [Đã triển khai v2].** Dự báo hằng số $\bar y_{\text{train}}$. Nếu tỷ lệ thắng của test bằng $\bar y$ thì Brier của nó bằng $\bar y(1-\bar y)$ (số hạng "không chắc chắn" trong phân rã Brier, mục 5.1) và Log Loss bằng entropy nhị phân của $\bar y$. Đây là sàn tối thiểu: mô hình phức tạp phải thắng được nó thì mới chứng minh học được tín hiệu.
* **`NumpyLogisticRegression` [Đã triển khai v2].** $\Pr(Y{=}1)=\sigma(w^\top\phi+b)$ trên đặc trưng tổng hợp `aggregate_history` (mean, std, min, max, last, slope theo từng tín hiệu, kèm ngữ cảnh và độ khó). Hàm mất mát log-loss lồi theo $(w, b)$, nên có nghiệm tối ưu duy nhất (với chính quy hóa). Hệ số chỉ so sánh được nếu đặc trưng đã chuẩn hóa. Vai trò: benchmark nhanh, giải thích được, phương án dự phòng.
* **`CandidateConditionedGRU` (v1) [Đã triển khai v2].** GRU trên chuỗi, độ khó và ngữ cảnh được nối (concatenate) trước lớp phân loại.
* **Baseline mạnh nên có:** cây tăng cường độ dốc (LightGBM/XGBoost) trên cùng đặc trưng tổng hợp là chuẩn so sánh mạnh cho dữ liệu dạng bảng. Rừng ngẫu nhiên (Random Forest) có thể dùng làm baseline hoặc công cụ phân tích; xem lưu ý hiệu chuẩn và diễn giải ở mục 5.1. **[Khuyến nghị]** Cả hai nằm ngoài danh sách mô-đun của v2.
* **Lập luận khoa học cần có:** nếu mô hình chuỗi không thắng rõ rệt Logistic/GBDT trên đặc trưng tổng hợp thì lợi ích của GRU chưa được chứng minh; đây là kết quả ablation bắt buộc báo cáo.

### 4.2. Gated Recurrent Unit (GRU)

* **Phương trình theo quy ước PyTorch** (cùng cách mã nguồn dùng `nn.GRU`), với đầu vào $\zeta_t$ (vector $z_t$ sau pooling, ký hiệu khác để không nhầm với logit):

$$\begin{aligned}
r_t &= \sigma\big(W_{ir}\zeta_t + b_{ir} + W_{hr}h_{t-1} + b_{hr}\big) &&\text{(cổng reset)}\\
u_t &= \sigma\big(W_{iu}\zeta_t + b_{iu} + W_{hu}h_{t-1} + b_{hu}\big) &&\text{(cổng cập nhật)}\\
n_t &= \tanh\!\big(W_{in}\zeta_t + b_{in} + r_t \odot (W_{hn}h_{t-1} + b_{hn})\big) &&\text{(trạng thái ứng viên)}\\
h_t &= (1-u_t)\odot n_t + u_t \odot h_{t-1} &&\text{(cập nhật)}
\end{aligned}$$

  Ở quy ước này $u_t$ lớn nghĩa là *giữ lại* ký ức cũ; $r_t$ nhỏ nghĩa là *bỏ qua* quá khứ khi tính ứng viên mới. Một số tài liệu (Chung và cs.) đặt vai trò của $u_t$ ngược lại; báo cáo cần nêu quy ước đã dùng. $h_0 = 0$, và $h_t \in (-1,1)^{64}$ vì là tổ hợp lồi của các giá trị $\tanh$ và trạng thái trước (tính chất hữu ích cho FiLM).
* **Quy mô.** Với đầu vào 64 và ẩn 64: $3(64\cdot 64 + 64\cdot 64 + 2\cdot 64) = 24\,960$ tham số.
* **Lý do chọn và giới hạn của lập luận.** Chuỗi ngắn ($L=12$), dữ liệu quy mô vừa, suy luận nhẹ và có trạng thái. Không nên khẳng định "GRU ít tham số hơn Transformer" như quy luật chung, vì phụ thuộc kích thước; lập luận vững là *thiên kiến quy nạp phù hợp, chi phí thấp, ngân sách runtime*. Cần ablation GRU so với LSTM, TCN (1D-CNN), Transformer nhỏ, và so với GBDT trên đặc trưng tổng hợp. **[Khuyến nghị]**

### 4.3. Tokenization theo từng đặc trưng và Masked Gated Weighted Pooling

**Kiến trúc [Đã triển khai v2]** (tại mỗi bước $t$, với đặc trưng $i\in\{0,\dots,20\}$):

$$e_{t,i} = \text{ReLU}\big(W_v[\tilde x_{t,i},\, m_{t,i}] + b_v\big) + E[i]\in\mathbb{R}^{32},\qquad q_{t,i} = \text{ReLU}(W_q e_{t,i} + b_q)\in\mathbb{R}^{64},$$

$$g_{t,i} = w_g^\top q_{t,i} + b_g,\qquad a_{t,i} = \exp\!\big(\tanh g_{t,i}\big)\, m_{t,i},\qquad \bar a_{t,i} = \frac{a_{t,i}}{\max\big(\sum_j a_{t,j},\,10^{-6}\big)},\qquad \zeta_t = \sum_i \bar a_{t,i}\, q_{t,i}.$$

Ở đây $E\in\mathbb{R}^{21\times 32}$ là bảng embedding danh tính đặc trưng. Đây là bản chất của *tokenizer theo đặc trưng* (cùng họ với FT-Transformer) và *pooling bất biến hoán vị kiểu DeepSets*: tập token có số lượng khả dụng thay đổi được gộp thành vector cố định, không phụ thuộc thứ tự vì danh tính nằm trong $E[i]$.

**Mệnh đề (tính chất của pooling).** Gọi $M_t=\sum_i m_{t,i}$ là số đặc trưng khả dụng tại $t$.

1. $a_{t,i}\in\{0\}\cup[e^{-1}, e]=\{0\}\cup[0.368,\,2.718]$, và tỉ số giữa hai trọng số khả dụng bất kỳ không quá $e^{2}\approx 7.39$. Pooling là *attention mềm có độ tương phản bị chặn*.
2. Nếu $M_t\ge 1$ thì $\sum_j a_{t,j}\ge M_te^{-1}\ge 0.368 > 10^{-6}$: **cận $10^{-6}$ không hoạt động**, $\sum_i\bar a_{t,i}=1$ chính xác, và $\zeta_t$ là tổ hợp lồi nên $\lVert\zeta_t\rVert\le\max_i\lVert q_{t,i}\rVert$.
3. Nếu $M_t=0$ (tất cả bị mask) thì $a\equiv 0$ và $\zeta_t=\mathbf 0$ (tổng trọng số bằng $0$, không phải $1$). Cận $10^{-6}$ chỉ có tác dụng ở trường hợp này: tránh chia cho 0 và tránh `NaN`.
4. $\zeta_t$ là **trung bình có trọng số**, nên **không mã hóa $M_t$** (số lượng đặc trưng khả dụng) cũng như nhóm nào đang thiếu. Thông tin này chỉ đến mô hình gián tiếp qua thành phần của token còn lại. Đề xuất nối thêm vector độ phủ 4 nhóm $(c_{\text{progress}}, c_{\text{vitality}}, c_{\text{capability}}, c_{\text{motor}})\in[0,1]^4$ vào đầu vào của GRU hoặc của head. **[Khuyến nghị]**

**Vì sao thay `MaskedFill(-inf)` + Softmax.** Khi tất cả vị trí bị mask, softmax trên toàn $-\infty$ cho `NaN`; còn dùng hằng số âm lớn (ví dụ $-10^9$) có thể tràn thành $-\infty$ trong FP16 (giá trị tối đa $65\,504$). Nhân mask tuyến tính vào trọng số bị chặn $\exp(\tanh(\cdot))$ tránh được cả hai, và đồ thị ONNX chỉ dùng toán tử cơ bản (`Tanh`, `Exp`, `Mul`, `ReduceSum`, `Div`, `Max`). Cái giá là độ tương phản trọng số bị chặn (mệnh đề 1).

**Tách biệt `0` thật và `thiếu`.** Cặp $(\tilde x,m)=(0,1)$ cho token có trọng số khác $0$, còn $(0,0)$ cho trọng số $0$: mô hình không bị nhầm giữa hai trường hợp.

### 4.4. FiLM, đầu ra và các kiểm tra theo độ khó

**Cấu trúc [Đã triển khai v2]** (với $h_L$ là trạng thái ẩn cuối của GRU):

$$\text{de}=\text{ReLU}(W_d\,d+b_d)\in\mathbb{R}^{32},\quad \gamma=W_\gamma\text{de}+b_\gamma,\quad \beta=W_\beta\text{de}+b_\beta,\quad h'=\gamma\odot h_{L}+\beta\in\mathbb{R}^{64},$$

$$\text{ctx}=\text{ReLU}(W_c\,c+b_c)\in\mathbb{R}^{16},\qquad z=W_2\,\text{ReLU}\!\big(W_1[\,h'\,\Vert\,\text{ctx}\,\Vert\,\text{de}\,]+b_1\big)+b_2.$$

Đầu vào của head có $64+16+32=112$ chiều. **Tổng số tham số (với $C=3$): 36 194**, phân bổ: tokenizer + pooling 2 945; GRU 24 960; nhánh độ khó và FiLM 4 576; ContextNet 64; head 3 649. Mô hình rất nhỏ (cỡ 36 nghìn), thuận lợi cho runtime nhưng cũng cho thấy rủi ro "thiếu dung lượng" nếu phải bao phủ nhiều game.

**Điều cần nêu chính xác trong báo cáo.**

1. **Vị trí của FiLM (nhất quán thời gian).** $d$ là *hành động tương lai*, nên không được đưa vào các bước thời gian của GRU: làm vậy khiến trạng thái quá khứ phụ thuộc vào quyết định chưa xảy ra và phá ngữ nghĩa $\operatorname{do}(d)$. Đặt FiLM sau GRU còn cho phép về nguyên tắc dùng một mã hóa lịch sử $h_L$ cho nhiều ứng viên.
2. **$d$ đi vào mạng hai lần** (qua FiLM và qua nối trực tiếp `de`), nên FiLM không phải đường duy nhất. Hợp lý, nhưng cần ablation để biết thành phần nào mang tác động.
3. **Khởi tạo.** Với $\gamma = W_\gamma\text{de}+b_\gamma$, tại thời điểm khởi tạo $\gamma\approx 0$ nên $h'\approx\beta$ và bỏ qua lịch sử. Tham số hóa chuẩn là $\gamma = \mathbf 1 + \Delta\gamma$ với lớp cuối khởi tạo gần $0$ (FiLM bắt đầu từ ánh xạ đồng nhất). **[Khuyến nghị]**
4. **Hiệu quả tính toán.** Tách bộ mã hóa (tokenizer + GRU) khỏi head thì chạy mã hóa một lần cho cả 3 ứng viên. Đồ thị ONNX hiện tại nhận $d$ là một đầu vào của cùng một đồ thị, nên 3 ứng viên $=$ 3 lượt chạy đầy đủ (xem mục 8.2).

**Kiểm tra độ nhạy theo $d$ (bắt buộc vì mạng là S-learner).** **[Khuyến nghị]** Với mỗi trạng thái $(h,c)$ ở tập validation, tính $p(d^-), p(d^0), p(d^+)$ cho ba ứng viên và báo cáo:

* *Biên độ phản hồi:* $\overline{|p(d^+)-p(d^-)|}$. Nếu xấp xỉ $0$, mạng đã bỏ qua $d$ và bộ điều khiển mù về lựa chọn.
* *Tỷ lệ vi phạm đơn điệu* $V=\frac1N\sum\mathbb 1[p(d^+)>p(d^0)+\epsilon]$, và tương quan thứ hạng giữa `relative_step` và $p$.
* *Hiệu chuẩn trong từng nhóm $d$* trên dữ liệu ngẫu nhiên hóa (nơi nhận dạng được tác động).

**Tính đơn điệu theo độ khó.** Trực giác thiết kế: nếu giữ nguyên $(h,c)$ và tăng độ khó theo mọi tọa độ thì xác suất thành công không tăng:
$$d' \succeq_{\text{khó}} d \;\Longrightarrow\; \mu(h,c,d')\le \mu(h,c,d),\qquad \mathcal{L}_{\text{mono}}=\max\big(0,\;p(d')-p(d)\big).$$
Với quy ước dấu thống nhất (mục 2.5), quan hệ thứ tự $\succeq_{\text{khó}}$ là thứ tự riêng theo từng tọa độ; khi các tọa độ ngược chiều (tăng áp lực nhưng tăng tiếp tế) thì *không có thứ tự*, nên chỉ kiểm tra đơn điệu dọc theo `relative_step`. Có thể đo $V$ **không cần nhãn**, bằng cách truy vấn mô hình trên các cặp $d,d'$. Không nên ép ràng buộc ngay từ đầu nếu có đánh đổi phi tuyến; ưu tiên đo $V$ rồi mới quyết định thêm $\mathcal{L}_{\text{mono}}$ hoặc dùng kiến trúc đơn điệu (lattice). **[Khuyến nghị/Mở rộng]**

### 4.5. Hàm mất mát, tối ưu hóa và điều chuẩn

* **BCE với logit [Đã triển khai v2].** $\ell(z,y) = -y\ln\sigma(z)-(1-y)\ln(1-\sigma(z)) = \ln(1+e^{-z})+(1-y)z$. Dạng ổn định số (log-sum-exp) là
$$\ell(z,y)=\max(z,0)-zy+\ln\!\big(1+e^{-|z|}\big),\qquad \frac{\partial\ell}{\partial z}=\sigma(z)-y.$$
  Kiểm tra: với $y=1$, $z\ge 0$ cho $\ln(1+e^{-z})$; $z<0$ cho $-z+\ln(1+e^{z})=\ln(1+e^{-z})$ (đã kiểm số học). Log-loss là *quy tắc chấm điểm chuẩn xác (proper scoring rule)*: cực tiểu kỳ vọng đạt tại xác suất thật, nên mạng huấn luyện bằng BCE hướng tới xác suất hiệu chuẩn, nhưng mạng quá khớp vẫn có thể quá tự tin (Guo và cs., 2017), do đó cần hiệu chuẩn sau huấn luyện (Phần 5).
* **Adam [Đã triển khai v2].**
$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\; v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\; \hat m_t=\frac{m_t}{1-\beta_1^t},\; \hat v_t=\frac{v_t}{1-\beta_2^t},\; \theta\leftarrow\theta-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$
* **L2 và weight decay không tương đương dưới Adam.** Cộng $\lambda\lVert\theta\rVert^2$ vào hàm mất mát khiến độ phạt bị chia cho mẫu số thích nghi $\sqrt{\hat v_t}$, nên không còn là suy giảm trọng số đồng đều; **AdamW** tách riêng: $\theta\leftarrow\theta-\eta\big(\hat m_t/(\sqrt{\hat v_t}+\epsilon)+\lambda\theta\big)$. Báo cáo cần nêu rõ dùng cái nào. **[Cần xác minh]**
* **Các biện pháp chống quá khớp (cấu hình tham chiếu, [Cần xác minh] giá trị trong mã):** Dropout $p=0.20$; weight decay $\lambda=10^{-5}$; dừng sớm `patience = 10` trên **tập validation** và khôi phục checkpoint tốt nhất (tuyệt đối không dùng test). Lưu ý:
  1. Trong `nn.GRU`, tham số `dropout` chỉ tác dụng **giữa các lớp xếp chồng**; GRU một lớp bỏ qua. Dropout phải đặt trên $\zeta_t$, $h_L$ hoặc head.
  2. Tiêu chí dừng nên là quy tắc chấm điểm chuẩn xác (log-loss hoặc Brier) trên validation, không phải accuracy.
  3. Clip giá trị $[-8,8]$ của normalizer là *chống ngoại lai/ổn định đầu vào*, không phải kỹ thuật chống quá khớp.
  4. Thêm *gradient clipping* (chuẩn, ví dụ 1.0) cho RNN và báo cáo trung bình ± độ lệch trên $\ge 3$–5 seed. **[Khuyến nghị]**

### 4.6. Mất cân bằng lớp

Trong dữ liệu thực, `segment_success=1` thường chiếm 80–90%; một mô hình luôn đoán $1$ đạt accuracy cao nhưng vô dụng.

* **Weighted BCE làm lệch xác suất.** Nếu loss gán trọng số $w_1$ cho $y=1$ và $w_0$ cho $y=0$, nghiệm tối ưu của tổng thể là $p_w=\dfrac{w_1p}{w_1p+w_0(1-p)}$, tức **logit bị cộng thêm $\ln(w_1/w_0)$**. (Đã kiểm số học: $w_1/w_0=3$ cho độ lệch $1.0986=\ln 3$.) Muốn có xác suất thật, phải hiệu chỉnh ngược:
$$z_{\text{true}} = z_w - \ln\frac{w_1}{w_0}.$$
* **Khuyến nghị mặc định [Khuyến nghị]:** huấn luyện **không trọng số**, vì đầu ra đi thẳng vào $\text{Score}(d)$ với $P_{\text{target}}$ và cần là xác suất hiệu chuẩn. Dùng class weight chỉ khi cần cải thiện biểu diễn, và khi đó bắt buộc hiệu chỉnh ngược trước khi hiệu chuẩn.
* **Thước đo phù hợp:** *Balanced Accuracy* $=\tfrac12(\text{TPR}+\text{TNR})$; recall của lớp *failure*; PR-AUC với failure là lớp dương (mục 5.1). Chi phí bất đối xứng (tăng nhầm nặng hơn giảm nhầm) nên đưa vào hàm quyết định $\text{Score}(d)$ (mục 6.2), không đưa vào class weight.
* **Toàn vẹn:** không oversample/SMOTE trên validation và test (phải giữ phân phối tự nhiên để ECE/Brier trung thực). SMOTE nội suy giữa các chuỗi sẽ tạo quỹ đạo không có thật, nên tránh cả trên train.

### 4.7. Mở rộng về mô hình **[Mở rộng]**

* **Học đa nhiệm:** $\mathcal{L}=\mathcal{L}_{\text{success}}+\alpha\mathcal{L}_{\text{death}}+\beta\mathcal{L}_{\text{progress}}+\gamma\mathcal{L}_{\text{time}}$. *Huber loss* $L_\delta(e)=\tfrac12e^2$ nếu $|e|\le\delta$, ngược lại $\delta(|e|-\tfrac12\delta)$ (như MSE khi sai số nhỏ, như MAE khi lớn). Hai lưu ý: (i) `time_to_checkpoint` bị **kiểm duyệt** khi phiên kết thúc sớm, nên Huber không phù hợp, cần mô hình thời gian-tới-sự-kiện (hazard rời rạc); (ii) cần cân bằng loss (uncertainty weighting, GradNorm) để tránh chuyển giao âm. v2 giữ đơn nhiệm để hội tụ ổn định trước.
* **Deep Ensemble:** xem mục 5.3.

---

## PHẦN 5: ĐÁNH GIÁ XÁC SUẤT, HIỆU CHUẨN VÀ BẤT ĐỊNH

### 5.1. Hệ thống thước đo

Đầu ra của mô hình được dùng như **xác suất** trong $\text{Score}(d)$, nên thước đo phải đánh giá chất lượng xác suất, không chỉ chất lượng phân loại.

* **Brier Score** $=\frac1N\sum_i(p_i-y_i)^2$, là quy tắc chấm điểm chuẩn xác, bị chặn trong $[0,1]$. **Phân rã Murphy:** $\text{Brier}=\text{REL}-\text{RES}+\text{UNC}$ với
$$\text{REL}=\sum_m\tfrac{n_m}{N}(\bar p_m-\bar y_m)^2\;(\text{độ sai hiệu chuẩn}),\quad \text{RES}=\sum_m\tfrac{n_m}{N}(\bar y_m-\bar y)^2\;(\text{độ phân biệt}),\quad \text{UNC}=\bar y(1-\bar y).$$
  Đẳng thức đúng chính xác khi dự báo nhận hữu hạn giá trị rời rạc, và xấp xỉ khi chia bin (đã kiểm số học). Phân rã cho biết Brier tốt lên do phân biệt tốt hơn hay do hiệu chuẩn tốt hơn.
* **Log Loss** (BCE) $=-\frac1N\sum_i[y_i\ln p_i+(1-y_i)\ln(1-p_i)]$, phạt nặng dự báo quá tự tin nhưng sai ($p\to0$ khi $y=1$ cho giá trị $\to\infty$, nên cần clip $p\in[\varepsilon,1-\varepsilon]$). Brier có mức phạt tối đa 1 mỗi mẫu; Log Loss không bị chặn.
* **ECE** $=\sum_{m=1}^M\frac{|B_m|}{N}\,\big|\text{freq}(B_m)-\text{conf}(B_m)\big|$, với $\text{freq}$ là tần suất thực $y=1$ trong bin và $\text{conf}$ là trung bình $p$ trong bin (với bài toán nhị phân, "acc" chính là $\text{freq}$ của lớp $1$). Giới hạn cần nêu: (i) phụ thuộc số bin và cách chia (bằng độ rộng hay bằng số mẫu); (ii) chệch lên khi mẫu nhỏ; (iii) *không* là quy tắc chấm điểm chuẩn xác; (iv) chỉ báo cáo kèm khoảng tin cậy. Vì bộ điều khiển chỉ vận hành quanh $P_{\text{target}}$, nên nên báo cáo thêm **hiệu chuẩn cục bộ trong $[0.6, 0.8]$**, cùng **độ dốc và hệ số chặn hiệu chuẩn** (hồi quy logistic của $y$ theo $z$: lý tưởng độ dốc $=1$, chặn $=0$). **[Khuyến nghị]**
* **Reliability Diagram (diễn giải chính xác).** Trục hoành là $p$ dự báo, trục tung là tần suất thực. Điểm **dưới đường chéo** nghĩa là mô hình *dự báo cao hơn thực tế* (tần suất thực $<p$): lạc quan về xác suất thành công. Điểm **trên đường chéo** là bi quan. *Quá tự tin* (cực đoan hơn thực tế) có dạng chữ S: dưới đường chéo ở $p$ cao và trên đường chéo ở $p$ thấp. Với DDA, lệch dưới đường chéo quanh $0.7$ nghĩa là mô hình đánh giá quá cao khả năng thắng, và bộ điều khiển có nguy cơ *tăng độ khó oan*.
* **ROC-AUC và PR-AUC.** Cả hai là thước đo *thứ hạng*, không đổi khi hiệu chuẩn bằng phép biến đổi đơn điệu (như nhiệt độ). Khi lớp thắng chiếm đa số, ROC-AUC dễ lạc quan vì $\text{FPR}=\text{FP}/(\text{FP}+\text{TN})$ có mẫu số lớn. Dùng **PR-AUC với failure ($y=0$) là lớp dương, điểm số $1-p$**; đường cơ sở của PR-AUC là *tỷ lệ failure* (không phải 0.5). Nếu lấy success làm lớp dương thì PR-AUC cao và vô nghĩa.
* **Độ tin cậy thống kê.** Mọi thước đo kèm khoảng tin cậy bằng *cluster bootstrap theo người chơi* (mục 3.2).

### 5.2. Hiệu chuẩn xác suất và Batch-MAP Temperature Scaling

**Temperature Scaling.** $\hat p=\sigma(z/\tau)=\sigma(\beta z)$ với $\beta=1/\tau$. $\tau>1$ làm mượt xác suất, $\tau<1$ làm sắc. Vì $u\mapsto\sigma(u/\tau)$ đơn điệu tăng nên **thứ hạng và dấu của logit không đổi**: ROC/PR-AUC và accuracy ở ngưỡng $0.5$ không đổi; chỉ độ tự tin đổi (và accuracy ở ngưỡng khác).

**Tối ưu MAP [Đã triển khai v2].** Buffer lưu cặp $(z_i,y_i)$ (lưu logit trực tiếp, tránh tính $\ln\frac p{1-p}$ lặp và mất chính xác khi $p\to0,1$); đủ 20 mẫu thì cập nhật, trên cửa sổ tối đa 60 mẫu gần nhất, 25 bước gradient, rồi kẹp $\tau\in[0.7,2.0]$:

$$J(\beta)=\sum_{i}\ell(\beta z_i,\,y_i)+\frac{\lambda}{2}(\beta-1)^2,\qquad \frac{\partial J}{\partial\beta}=\sum_i\big(\sigma(\beta z_i)-y_i\big)z_i+\lambda(\beta-1).$$

* **Tính lồi.** $\ell(\cdot,y)$ lồi theo logit, nên $\sum\ell(\beta z_i,y_i)$ lồi theo $\beta$ (hợp với phép tuyến tính); thêm số hạng bậc hai thì $J$ lồi chặt, nghiệm duy nhất. Nếu tối ưu theo $\tau$ trực tiếp thì NLL nói chung **không lồi** theo $\tau$. Khuyến nghị tối ưu theo $\beta$ (hoặc $\ln\tau$), rồi kẹp. Khoảng $\tau\in[0.7,2.0]$ tương ứng $\beta\in[0.5,\,1.4286]$ và $\ln\tau\in[-0.357,\,0.693]$: không đối xứng theo thang log, cần nêu lý do. **[Khuyến nghị]**
* Sơ đồ kiến trúc viết phạt là $\lambda(\tau-\tau_{\text{base}})$ trong khi văn bản viết $\tfrac12\lambda(\tau-\tau_{\text{base}})^2$; hai biểu thức nhất quán nếu cái đầu là *gradient* của cái sau.

**Ba giới hạn lý thuyết quan trọng.**

1. **Nhiệt độ không sửa được lệch (bias).** Nếu mọi $z=0$ thì $\sigma(\beta z)=0.5$ với mọi $\tau$. Khi tỷ lệ thắng nền đổi (label shift, mục 3.4), cần phép *dịch* logit, không phải phép co giãn.
2. **Logit tập trung làm $\tau$ và bias không tách được.** Bộ điều khiển đưa $p$ về quanh $0.70$, nên các logit trong buffer tập trung quanh $z_0=\ln(7/3)=0.847$. Khi đó likelihood phụ thuộc $\beta$ chỉ qua tích $\beta z_0$, và nghiệm MLE là $\hat\beta=\operatorname{logit}(\bar y)/z_0$. Ví dụ: tỷ lệ thắng quan sát $\bar y=0.80$ cho $\hat\beta=1.636$, $\hat\tau=0.611$ (bị kẹp về $0.7$). Phép hiệu chỉnh này *làm sắc* toàn bộ dự báo: $p=0.9\to0.973$ và $p=0.2\to0.094$ (với $\tau=0.611$), trong khi nếu lệch thật chỉ là *dịch* thì $p=0.9\to0.939$ và $p=0.2\to0.300$. Cả hai đều đưa $0.7\to0.8$ nhưng cho kết quả khác hẳn ở nơi khác. Hệ quả: khi $\operatorname{Var}(z)$ trong buffer nhỏ, chỉ có độ lệch tại điểm vận hành là nhận dạng được.
   **Đề xuất [Khuyến nghị]:** dùng *Platt scaling có tiên nghiệm* $\hat p=\sigma(az+b)$ với phạt $\tfrac{\lambda_a}{2}(a-1)^2+\tfrac{\lambda_b}{2}b^2$ (vẫn lồi chung theo $(a,b)$); và khi $\operatorname{Var}(z)$ dưới ngưỡng, chỉ cập nhật $b$ và giữ $a$. Isotonic regression và beta calibration cần nhiều mẫu hơn (hàng trăm), không phù hợp buffer 20–60 mẫu.
3. **Tính độc lập và phụ thuộc chính sách của mẫu.** Các segment liên tiếp của một người chơi tự tương quan, nên $n_{\text{eff}}<60$; mẫu thu theo chính sách điều khiển (on-policy) tập trung ở các độ khó được chọn; một $\tau$ dùng chung cho cả 3 ứng viên (giả định lệch đồng đều). Nhãn có độ trễ (biết sau khi phân đoạn kết thúc) và quy tắc nhãn `quit` phải trùng với lúc huấn luyện (mục 1.4).

**Ảnh hưởng tới quyết định.** Phép đổi đơn điệu không đổi thứ hạng giữa các ứng viên, nhưng $\text{Score}$ dùng $|\hat p-P_{\text{target}}|$ và các cổng dùng ngưỡng trên $\hat p$, nên hiệu chuẩn **có làm đổi quyết định**.

**Kiểm thử đề xuất.** (i) Test đơn vị tổng hợp: sinh $z\sim\mathcal N$, nhãn từ $\sigma(\beta^\star z)$ với $\beta^\star$ biết trước, kiểm tra thuật toán khôi phục $\beta^\star$ khi $n$ đủ lớn; (ii) NLL trên buffer không tăng sau cập nhật; (iii) $\tau$ luôn trong biên; (iv) Leave-Session-Out cho bậc 2: so sánh $\tau=1$ (zero-shot) và $\tau$ đã hiệu chuẩn (few-shot) trên ECE, Brier, Log Loss. **[Khuyến nghị]**

### 5.3. Bất định: Aleatoric, Epistemic và độ tin cậy theo độ phủ đầu vào

* **Aleatoric:** ngẫu nhiên vốn có của trò chơi (xúc xắc, hành vi người chơi), không giảm được bằng thêm dữ liệu. **Epistemic:** thiếu hiểu biết của *mô hình* (ít dữ liệu, ngoài phân phối), giảm được bằng thêm dữ liệu.

**Độ tin cậy theo độ phủ đầu vào (`estimate_mask_uncertainty`) [Đã triển khai v2].** Gọi $c_g\in[0,1]$ là tỷ lệ biến khả dụng của nhóm $g$ ($|g|=4,4,4,9$):

$$\text{coverage}=\tfrac14\sum_{g=1}^4c_g,\qquad \mathcal{U}=u_{\text{base}}+(1-\text{coverage})(1-u_{\text{base}}),\quad u_{\text{base}}=0.05,$$
$$\text{Confidence}=1-\mathcal{U}=(1-u_{\text{base}})\cdot\text{coverage}=0.95\cdot\text{coverage}.$$

Cổng bất định yêu cầu $\text{Confidence}\ge0.80\iff\text{coverage}\ge0.8421$. Hai ví dụ trong thiết kế khớp (đã kiểm): thiếu 3 biến lịch sử của nhóm 9 biến: coverage $=0.25(3+6/9)=0.9167$, $\mathcal U=0.1292$, Confidence $=0.8708\ge0.80$ (bộ điều khiển chạy); mất cả một nhóm: coverage $=0.75$, Confidence $=0.7125<0.80$ (`HOLD_LOW_CONFIDENCE`).

**Bảng ngưỡng khi chỉ một nhóm bị thiếu (các nhóm khác đầy đủ).**

| Nhóm | Số biến mất | Coverage | Confidence | Kết quả |
|---|:---:|:---:|:---:|:---:|
| 4 biến | 0 / 1 / 2 | 1.000 / 0.938 / 0.875 | 0.950 / 0.891 / 0.831 | Chạy |
| 4 biến | 3 / 4 | 0.813 / 0.750 | 0.772 / 0.713 | `HOLD` |
| 9 biến (`human_cognitive_motor`) | $\le5$ | $\ge0.861$ | $\ge0.818$ | Chạy |
| 9 biến (`human_cognitive_motor`) | $\ge6$ | $\le0.833$ | $\le0.792$ | `HOLD` |

Điều kiện tổng quát khi chỉ một nhóm thiếu: $c_g\ge0.368$. Cần nhớ: hệ thống chịu mất tới 5/9 biến của nhóm cuối nhưng chỉ chịu mất 2/4 biến của các nhóm khác, hệ quả trực tiếp của trọng số nhóm đều nhau; nên biện minh bằng ablation (mục 7.1) hoặc dùng trọng số theo tầm quan trọng.

**Ba điểm cần nêu chính xác:**

1. **Đây không phải epistemic uncertainty theo nghĩa mô hình.** Nó đo *độ phủ dữ liệu đầu vào* và nên gọi là *Input-Coverage Confidence*. Nó không phát hiện được đầu vào ngoài phân phối khi mọi đặc trưng đều có mặt. Epistemic thật cần ensemble, MC-Dropout, hoặc phát hiện OOD. **[Khuyến nghị đổi tên]**
2. **Không phụ thuộc $d$.** Vì chỉ là hàm của mặt nạ, $\mathcal{U}(d)\equiv\mathcal{U}$ là hằng số trên các ứng viên, nên số hạng $\lambda_u\mathcal{U}(d)$ trong $\text{Score}(d)$ **không ảnh hưởng đến `argmax`** mà chỉ có tác dụng qua cổng Confidence. Muốn "né rủi ro" thật sự, dùng $\mathcal{U}(d)=\mathcal{U}_{\text{cov}}+\mathcal{U}_{\text{model}}(d)$, với $\mathcal{U}_{\text{model}}(d)$ là độ bất đồng của ensemble hoặc khoảng cách của $d$ tới vùng được huấn luyện. **[Khuyến nghị]**
3. **Ví dụ động cơ trong thiết kế cần thống nhất với ngưỡng.** Trung bình phẳng trên 21 biến khi thiếu 3 biến cho coverage $18/21=0.857$, Confidence $=0.814\ge0.80$, vẫn vượt ngưỡng; do đó khẳng định "gây khóa cứng DDA" chỉ đúng với ngưỡng cao hơn (khoảng $0.85$). Lợi ích của trọng số nhóm cần được nêu bằng ví dụ khác (ví dụ phân biệt mất một nhóm cấu trúc với mất vài biến lịch sử).

**Entropy và ensemble.** Với $K$ mô hình có $p_k$, $\bar p=\frac1K\sum p_k$:
$$\underbrace{H[\bar p]}_{\text{tổng}}=\underbrace{\tfrac1K\textstyle\sum_kH[p_k]}_{\text{aleatoric (kỳ vọng entropy)}}+\underbrace{H[\bar p]-\tfrac1K\textstyle\sum_kH[p_k]}_{\text{epistemic (thông tin tương hỗ)}\ \ge 0},\qquad H[p]=-p\ln p-(1-p)\ln(1-p).$$
Bất đẳng thức do tính lõm của $H$ (Jensen). Entropy cực đại $\ln2=0.693$ tại $p=0.5$, và $H(0.7)=0.611$, bằng $88\%$ mức cực đại. Vì hệ thống *cố ý vận hành ở vùng entropy cao*, entropy dự báo đơn lẻ không phân biệt được "đang ở mục tiêu" với "đang bất định"; nó cũng không tăng khi mô hình đoán sai một cách tự tin. Thành phần *thông tin tương hỗ* của ensemble mới là tín hiệu epistemic hữu ích (tốn $K$ lần tính toán, phù hợp offline). **[Mở rộng]**

### 5.4. Chỉ số khó khăn (Struggle Index)

**Định nghĩa [Đã triển khai v2].** `estimate_struggle_probability` là một **heuristic**:

$$P_{\text{struggle}}=\begin{cases}1-p(d_{\text{cur}}), & \text{win rate chưa khả dụng},\\[2pt] 0.6\,\big(1-p(d_{\text{cur}})\big)+0.4\,\big(1-\text{recent\_win\_rate}\big), & \text{win rate khả dụng}.\end{cases}$$

Việc dùng $p(d_{\text{cur}})$ (không dùng trung bình các ứng viên) phản ánh đúng mức hiện tại; kiểm tra khả dụng ngăn giá trị giữ chỗ $0$ của `recent_win_rate` cộng nhầm $0.4$ điểm ở phân đoạn đầu. $p$ nên là **xác suất đã hiệu chuẩn** $\hat p$ **[Cần xác minh]**.

**Phân tích lý thuyết.**

1. Là tổ hợp lồi của hai đại lượng trong $[0,1]$ nên nằm trong $[0,1]$, nhưng **không phải xác suất của một biến cố được định nghĩa**; nên gọi là *chỉ số* (index), trọng số $0.6/0.4$ là tham số thiết kế.
2. **Đếm hai lần.** `recent_win_rate` đã là đầu vào của mạng (tín hiệu 18) nên $p$ đã chứa thông tin đó; kết hợp lại tạo sai số tương quan.
3. **Phương sai cao.** Với $k$ phân đoạn gần nhất, sai số chuẩn của tỷ lệ thắng là $\sqrt{\bar y(1-\bar y)/k}$, ví dụ $k=3$, $\bar y=0.5$ cho $0.29$.
4. **Gián đoạn** khi cờ khả dụng đổi từ $0$ sang $1$ (định nghĩa nhảy giữa hai nhánh). Có thể làm liên tục bằng trọng số tăng dần theo số phân đoạn quan sát: $w_r=0.4\,\dfrac{n_{\text{seg}}}{n_{\text{seg}}+\kappa_r}$, $P_{\text{struggle}}=(1-w_r)(1-\hat p)+w_r(1-\bar y)$. **[Khuyến nghị]**
5. Phương án đơn giản và có ý nghĩa rõ nhất: $P_{\text{struggle}}=1-\hat p(d_{\text{cur}})$, tức *xác suất thất bại phân đoạn kế* đã hiệu chuẩn; khi đó ngưỡng $0.75$ tương đương $\hat p(d_{\text{cur}})\le0.25$. Ngưỡng nên chọn bằng phân tích chi phí–lợi ích (ROC/chi phí) trên dữ liệu ngẫu nhiên hóa, không chọn tùy ý.

---

## PHẦN 6: ĐIỀU KHIỂN AN TOÀN VÀ LÝ THUYẾT TRẢI NGHIỆM NGƯỜI CHƠI

### 6.1. Kiến trúc tách rời và điều khiển vòng kín

* **Tách rời (decoupled):** mô hình học sâu chỉ *ước lượng* $\hat p(d)$; bộ điều khiển (`controller.py`) *ra quyết định* bằng logic xác định, kiểm tra được và có thể dịch sang C#. Lợi ích: kiểm thử đơn vị từng khối, khóa an toàn không phụ thuộc mạng, thay thế chính sách (rule, bandit) mà không đổi mô hình. **[Đã triển khai v2]**
* **Vòng kín.** Người chơi là một phần của hệ thống: độ khó $\to$ hành vi $\to$ telemetry $\to$ $\hat p$ $\to$ độ khó. Ba đặc điểm khiến bài toán khác điều khiển cơ học thông thường:
  1. **Trễ phản hồi:** nhãn chỉ biết sau khi phân đoạn (30–60 s) kết thúc, và quyết định chỉ áp dụng tại checkpoint (*lấy mẫu theo sự kiện*, không theo chu kỳ cố định).
  2. **Đối tượng không dừng (non-stationary):** người chơi học và thay đổi kỹ năng, nên quan hệ $d\mapsto p$ trôi theo thời gian.
  3. **Đo lường nhiễu:** $\hat p$ có sai số hiệu chuẩn và sai số thống kê, nên cần vùng chết và xác nhận liên tiếp để tránh phản ứng với nhiễu.
* **Hệ quả:** điều khiển quá nhạy gây dao động (thrashing); quá chậm gây bỏ lỡ trạng thái flow. Lý do hệ thống ưu tiên *thận trọng*: bước nhảy tối đa 1 profile, cooldown, vùng chết, xác nhận hai lần.

### 6.2. Hàm điểm ứng viên (Candidate Scoring)

Tập ứng viên hợp lệ: $\mathcal{D}(d_{\text{cur}})=\{d_{\text{cur}}-1,\,d_{\text{cur}},\,d_{\text{cur}}+1\}\cap\{0,\dots,D_{\max}\}$.

$$\text{Score}(d)=-\big|\hat p(d)-P_{\text{target}}\big|-\lambda_u\,\mathcal{U}(d)-\lambda_c\,|d-d_{\text{cur}}|,\qquad P_{\text{target}}=0.70,\;\lambda_u=0.25,\;\lambda_c=0.05.$$

* Đây là **vô hướng hóa trọng số** (weighted-sum scalarization) của ba mục tiêu: bám mục tiêu, né bất định, ít thay đổi. Phương pháp này không tìm được các điểm Pareto thuộc phần không lồi; chấp nhận được với 3 mục tiêu đơn giản.
* **Ý nghĩa của $\lambda_c$.** Chuyển một bậc chỉ đáng khi khoảng cách tới mục tiêu giảm quá $\lambda_c=0.05$ (cộng chênh lệch bất định). Ví dụ: $\hat p(d_{\text{cur}})=0.80$ (khoảng cách $0.10$, điểm $-0.10$), $\hat p(d_{\text{cur}}+1)=0.72$ (khoảng cách $0.02$, điểm $-0.02-0.05=-0.07$) thì tăng độ khó được chọn.
* **$\mathcal{U}(d)$ hằng số.** Như mục 5.3, nếu $\mathcal{U}$ chỉ phụ thuộc mặt nạ thì số hạng $\lambda_u\mathcal{U}$ bị triệt tiêu khi so sánh ứng viên. **[Khuyến nghị]** làm nó phụ thuộc $d$.
* **Mất mát bất đối xứng.** Thay $|\hat p-P_{\text{target}}|$ bằng $a\,(P_{\text{target}}-\hat p)_+ + b\,(\hat p-P_{\text{target}})_+$ với $a>b$ nếu coi "quá khó" nghiêm trọng hơn "quá dễ" (giả thuyết thiết kế, cần kiểm chứng bằng dữ liệu thoát/giữ chân). **[Khuyến nghị]**

### 6.3. Máy trạng thái quyết định và các bất biến an toàn

**Luồng quyết định tại mỗi checkpoint [Đã triển khai v2]:**

1. **Cooldown:** còn thời gian hồi $\Rightarrow$ `HOLD_COOLDOWN`; $\text{cooldown}\leftarrow\max(0,\text{cooldown}-\Delta t_{\text{real}})$ với $\Delta t_{\text{real}}$ là giây thực trôi qua (`dt_seconds`), nên cooldown 60 s kết thúc sau 1–2 checkpoint cách nhau 30–60 s, không bị kéo giãn thành hàng chục phút như trừ hằng số 1.0 mỗi lượt.
2. **Safe checkpoint:** không an toàn $\Rightarrow$ `HOLD_UNSAFE_CHECKPOINT` (không đổi giữa giao tranh).
3. **Confidence:** $<0.80\Rightarrow$ `HOLD_LOW_CONFIDENCE`.
4. **Phát hiện trạng thái:** $P_{\text{struggle}}\ge0.75\Rightarrow$ tăng `decrease_streak`, đặt lại `increase_streak`; "sẵn sàng tăng" (hai ước lượng $\hat p(d_{\text{cur}})$ và $\hat p(d_{\text{cur}}+1)$ đạt chuẩn, `progress` $\ge0.30$) $\Rightarrow$ tăng `increase_streak`; còn lại: đặt lại cả hai.
5. **Xác nhận:** streak $\ge2\Rightarrow$ áp dụng `DECREASE_STRUGGLE` ($\max(0,d_{\text{cur}}-1)$) hoặc `INCREASE_READY` ($d_{\text{cur}}+1$), rồi bật cooldown 60 s.

**Điểm cần làm rõ trong đặc tả [Cần xác minh]:**

* Ngưỡng số cụ thể của "đạt chuẩn" cho $\hat p(d_{\text{cur}})$ và $\hat p(d_{\text{cur}}+1)$ (ví dụ minh họa: $\hat p(d_{\text{cur}})\ge0.80$ và $\hat p(d_{\text{cur}}+1)\ge0.70$).
* **Vai trò của $\text{Score}$.** Sơ đồ tính $\text{Score}$ cho từng ứng viên nhưng hướng tăng/giảm lại được quyết bằng các ngưỡng (struggle, sẵn sàng tăng). Cần nêu rõ $\text{Score}$ dùng ở đâu (ví dụ: chọn $d^\star=\arg\max\text{Score}$ để xác định hướng, rồi các cổng và streak xác nhận hướng đó; $P_{\text{struggle}}$ là cơ chế ghi đè cho trường hợp khẩn).
* Trạng thái streak có bị đặt lại khi một cổng `HOLD` chặn hay không.
* **Thứ tự cập nhật cooldown:** nên trừ `dt_seconds` *trước* khi kiểm tra; nếu kiểm tra trước rồi mới trừ, cooldown còn 30 s với $\Delta t=40$ s vẫn trả `HOLD_COOLDOWN`. Giới hạn trên $d_{\text{cur}}+1\le D_{\max}$ cũng cần được kiểm tra.

**Các bất biến an toàn (dùng làm test thuộc tính/property-based test).** **[Khuyến nghị]**

| # | Bất biến |
|---|---|
| I1 | $|d_{\text{new}}-d_{\text{cur}}|\le1$ và $0\le d_{\text{new}}\le D_{\max}$ |
| I2 | Còn cooldown (sau khi trừ $\Delta t$) $\Rightarrow d_{\text{new}}=d_{\text{cur}}$ |
| I3 | Không phải safe checkpoint $\Rightarrow d_{\text{new}}=d_{\text{cur}}$ |
| I4 | $\text{Confidence}<0.80\Rightarrow d_{\text{new}}=d_{\text{cur}}$ |
| I5 | Đổi độ khó chỉ khi cùng một hướng được xác nhận $\ge2$ lần liên tiếp |
| I6 | Sau mỗi lần đổi: cooldown $=60$ s, cả hai streak $=0$ |
| I7 | Đầu ra mô hình thiếu/`NaN`/ngoài $[0,1]$ $\Rightarrow$ `HOLD` |
| I8 | Mọi cổng thất bại đều dẫn tới *không đổi* (fail-safe) |

Đây là một dạng **lá chắn an toàn (shield)** theo nghĩa của học tăng cường an toàn: tầng quy tắc luôn có quyền chặn đầu ra của tầng ước lượng.

### 6.4. Vùng chết, xác nhận liên tiếp và khóa thời gian: gọi đúng tên

| Cơ chế trong hệ thống | Tên kỹ thuật chính xác | Tác dụng |
|---|---|---|
| $P_{\text{struggle}}\ge0.75$ để giảm; "sẵn sàng tăng" để tăng; vùng ở giữa: giữ nguyên | **Vùng chết (dead band)** trên trục xác suất | Không phản ứng với dao động nhỏ quanh mục tiêu |
| Cần $\ge2$ chu kỳ liên tiếp | **Xác nhận liên tiếp (debouncing / persistence filter)** | Lọc nhiễu xung đơn lẻ |
| Cooldown 60 s theo thời gian thực | **Khóa (lockout / refractory period)** | Giới hạn tần suất đổi |

**Hysteresis theo nghĩa chặt** (kiểu Schmitt trigger) là ngưỡng *phụ thuộc trạng thái*: ví dụ ngưỡng giảm tụt thấp hơn ngay sau khi vừa tăng. Hệ thống chưa dùng điều này; nên dùng thuật ngữ "dead band + debounce + lockout" thay cho "hysteresis". **[Khuyến nghị đổi thuật ngữ]**

**Khoảng cách tối thiểu giữa hai lần đổi.** Giả sử checkpoint đều đặn mỗi $T_{cp}$ giây và streak chỉ đếm sau khi hết cooldown:
$$T_{\min}=\Big(\big\lceil T_{cd}/T_{cp}\big\rceil+N_{\text{streak}}-1\Big)\,T_{cp}.$$
Với $T_{cd}=60$, $N_{\text{streak}}=2$: $T_{cp}=30\Rightarrow90$ s; $T_{cp}=45\Rightarrow135$ s. Vùng chết nên rộng hơn độ lệch chuẩn của nhiễu ước lượng $\hat p$ thì mới có tác dụng chống dao động.

### 6.5. Chỉ số hiệu năng của bộ điều khiển

Gọi $\bar y^{(m)}_k$ là **tỷ lệ thắng trượt** trên $m$ phân đoạn gần nhất (ví dụ $m=5$ hoặc $10$). Kết quả từng phân đoạn là nhị phân nên không thể so trực tiếp với $0.70$.

* **Time-in-Band:** tỷ lệ phân đoạn có $\bar y^{(m)}_k\in[0.60,0.80]$ (dải mục tiêu).
* **Mean Target Deviation:** $\frac1K\sum_k|\bar y^{(m)}_k-P_{\text{target}}|$.
* **Tần suất đổi và dao động:** số lần đổi mỗi giờ; số lần *đảo chiều* (tăng rồi giảm trong cửa sổ $W$ phút).
* **False Increase Rate (FIR):** $\dfrac{\#\{\text{lần tăng mà trong } w \text{ phân đoạn kế tiếp xảy ra struggle}\}}{\#\{\text{lần tăng}\}}$, trong đó *struggle* phải được định nghĩa trước (ví dụ thua $\ge2$ phân đoạn liên tiếp trong $w=3$). Vì struggle cũng xảy ra không do tăng độ khó, cần báo cáo **mức dư so với đường cơ sở** (tỷ lệ struggle sau các quyết định `HOLD` hoặc ở nhánh Fixed trong trạng thái tương đương). **False Decrease Rate** định nghĩa đối xứng với biến cố "chán" (ví dụ chuỗi thắng dài với biên độ cao).
* **Chỉ số hậu quả:** retention (ngày kế), độ dài phiên, tỷ lệ thoát giữa phân đoạn, ý định tiếp tục chơi (khảo sát).
* **Regret so với Oracle** (chỉ tính offline trên dữ liệu ngẫu nhiên hóa, nơi có đủ kết quả cho mọi ứng viên).

### 6.6. Lý thuyết trải nghiệm: tính tự chủ, công bằng và rubber-banding

* **Rubber-banding** xuất phát từ cơ chế "kéo cân bằng" (catch-up) trong game đua xe: tốc độ AI bám theo vị trí người chơi. Nếu người chơi nhận ra độ khó đổi theo mình, trải nghiệm có thể bị giảm (cảm giác bị can thiệp). Vì vậy hệ thống giới hạn bước nhảy, tăng cooldown và chỉ đổi tại điểm an toàn.
* **Self-Determination Theory** (Ryan & Deci): thỏa mãn *năng lực (competence)*, *tự chủ (autonomy)* và *gắn kết (relatedness)* dự báo động lực nội tại. DDA hướng tới *competence*; điều chỉnh âm thầm có thể làm giảm *autonomy* cảm nhận. Đây là **giả thuyết cần kiểm chứng** bằng thang PENS, không phải kết luận.
* **Đo lường:** GEQ (Game Experience Questionnaire), FSS-2 (Flow State Scale), PENS, NASA-TLX (tải nhận thức), cộng câu hỏi *kiểm tra thao tác* (manipulation check) về mức khó cảm nhận (mục 7.3).

---

## PHẦN 7: THIẾT KẾ THỰC NGHIỆM VÀ KIỂM CHỨNG

### 7.1. Thực nghiệm ngoại tuyến

**Quy ước chung.** Mọi chia tập theo người chơi (mục 3.2); mọi thước đo ở mục 5.1 kèm khoảng tin cậy cluster bootstrap; báo cáo trung bình $\pm$ độ lệch trên $\ge3$ seed; mô hình so sánh: `MajorityModel`, Logistic, GBDT (nếu có), `CandidateConditionedGRU` tĩnh, `TokenizedFiLMGRU` (zero-shot) và `TokenizedFiLMGRU + OnlineTemperatureCalibrator` (few-shot).

**Benchmark A: khái quát hóa Cross-Map / Cross-Patch [Đã triển khai v2].** Huấn luyện trên Map 1/Patch v1; kiểm thử không đổi trọng số trên Map 2/Patch v2 (zero-shot, $\tau=1$, `empirical_prior` mask $=0$); sau đó few-shot bằng Batch-MAP theo giao thức *Leave-Session-Out* (hiệu chuẩn trên các phiên khác của miền đích, báo cáo trên phiên giữ lại). So sánh ECE (toàn cục và cục bộ), Brier (kèm phân rã REL/RES), Log Loss, PR-AUC của lớp failure. Mục tiêu giả thuyết: few-shot cải thiện hiệu chuẩn nhưng *không* cải thiện thứ hạng (AUC), nếu AUC giảm đáng kể thì nguyên nhân là concept drift (mục 3.4).

**Benchmark B: chịu lỗi rụng đặc trưng [Đã triển khai v2].** Che lần lượt từng nhóm trong 4 nhóm (và các mức che một phần). Tiêu chí:

1. Không crash, không `NaN`, đầu ra trong $[0,1]$.
2. Confidence tính ra **khớp công thức** $0.95\cdot\text{coverage}$ (test đơn vị bằng đẳng thức, không chỉ "có tăng").
3. Bộ điều khiển chuyển `HOLD_LOW_CONFIDENCE` đúng ngưỡng (bảng ở mục 5.3), không tăng độ khó bừa bãi.
4. **Thêm [Khuyến nghị]:** đường cong suy giảm hiệu năng (Brier/ECE theo tỷ lệ biến bị che) và so sánh mô hình *huấn luyện có feature-group dropout* với *không có* (mục 2.3). Nếu che một nhóm hầu như không đổi Brier thì trọng số $25\%$ của nhóm đó đang đánh giá quá cao tầm quan trọng của nó.

**Các thực nghiệm bổ sung cần có để mô hình "đúng về mặt lý thuyết" [Khuyến nghị]:**

| Mã | Nội dung | Liên quan |
|---|---|---|
| C | Độ nhạy theo $d$ và tỷ lệ vi phạm đơn điệu (không cần nhãn), hiệu chuẩn theo từng $d$ trên dữ liệu ngẫu nhiên hóa | Mục 4.4 |
| D | Ablation chuẩn hóa: (i) chỉ prior game, (ii) chỉ thống kê phiên, (iii) co cụm | Mục 2.6 |
| E | Đánh giá chỉ trên cửa sổ căn chỉnh theo điểm quyết định | Mục 3.5 |
| F | Test đơn vị bộ hiệu chuẩn: khôi phục $\beta^\star$ biết trước; so sánh $\tau$-only với Platt $(a,b)$ | Mục 5.2 |
| G | Ablation kiến trúc: mean pooling và gated pooling; FiLM và nối; GRU và LSTM/TCN/GBDT; có/không `empirical_prior` | Mục 4.2–4.4 |
| H | Mô phỏng vòng kín (mục 7.2) | Mục 6 |

### 7.2. Mô phỏng vòng kín với người chơi giả lập

Trước khi thử với người thật, kiểm tra riêng *logic điều khiển* bằng mô hình người chơi có đáp án biết trước. Ví dụ:
$$p^\star_k=\sigma\big(\theta_k-b(d_k)\big),\quad b(d)=b_0+\kappa d,\quad \theta_{k+1}=\theta_k+\eta+\xi_k,$$
với $\theta$ là năng lực (tăng dần do học), $\kappa$ là logit mỗi profile, $\xi$ là nhiễu. Cho bộ điều khiển nhận $\hat p=\sigma\big(\operatorname{logit}(p^\star)+\text{bias}+\text{noise}\big)$ với các mức bias/nhiễu khác nhau để đo độ bền vững. Vì biết $p^\star$, tính được chính xác Time-in-Band, số lần dao động, regret, và kiểm tra ảnh hưởng của trễ, độ rộng vùng chết, $\lambda_c$, $P_{\text{target}}$. Đây cũng là cách phát hiện lỗi thiết kế (dao động chu kỳ, kẹt ở biên) trước khi tốn chi phí thử nghiệm người thật. **[Khuyến nghị]**

### 7.3. Thử nghiệm có đối chứng ngẫu nhiên với người chơi

* **Các nhánh (arms):** (1) *Fixed* (ví dụ profile 2); (2) *Rule-based*; (3) *Model-based* (DDA-models); (4) *Sham-adaptive* (đổi độ khó ngẫu nhiên với cùng tần suất, để tách hiệu ứng "có thay đổi" khỏi hiệu ứng "thay đổi đúng"). *Oracle* chỉ là cận trên ngoại tuyến, không chơi trực tiếp.
* **Thiết kế:** trong-đối-tượng (crossover) với phân bổ thứ tự bằng Latin square (3–4 nhánh: thiết kế cân bằng bậc 1 như Williams), có thời gian rửa trôi (washout) để giảm hiệu ứng dư (carry-over) hoặc thiết kế giữa-đối-tượng nếu tác động học tập lớn. Người chơi không biết mình ở nhánh nào (mù đơn).
* **Giả thuyết đề xuất (khớp lại với bản H1–H5 của bạn, vì tài liệu kiến trúc không liệt kê):**

| Mã | Giả thuyết |
|---|---|
| H1 | Model-based có Time-in-Band cao hơn Fixed |
| H2 | Model-based có Time-in-Band cao hơn và ít dao động hơn Rule-based |
| H3 | Model-based có False Increase Rate (mức dư) thấp hơn Rule-based |
| H4 | Độ công bằng và tự chủ cảm nhận (PENS/GEQ) của Model-based **không kém hơn** Fixed (kiểm định không thua kém, biên đặt trước) |
| H5 | Ý định tiếp tục chơi/retention của Model-based không thấp hơn các nhánh còn lại |

* **Công cụ đo:** GEQ, FSS-2, PENS, NASA-TLX; câu kiểm tra thao tác (người chơi có cảm nhận khác biệt độ khó không); chỉ số hành vi từ telemetry (mục 6.5).

### 7.4. Thống kê cho thực nghiệm

* **Tiền đăng ký (pre-registration):** nêu trước giả thuyết, chỉ số chính, quy tắc loại dữ liệu, phương pháp phân tích.
* **Mô hình hỗn hợp tuyến tính** với hiệu ứng ngẫu nhiên theo người chơi: $y_{ij}=\beta_0+\beta_{\text{arm}(ij)}+u_i+\varepsilon_{ij}$, $u_i\sim\mathcal N(0,\sigma_u^2)$. Dữ liệu thứ bậc hoặc lệch dùng Friedman và Wilcoxon có hiệu chỉnh Holm.
* **Đa so sánh:** hiệu chỉnh Holm hoặc FDR cho H1–H5.
* **Cỡ hiệu ứng và khoảng tin cậy** (Cohen's $d_z$ hoặc rank-biserial), không chỉ $p$-value.
* **Công suất a priori** (xấp xỉ chuẩn cho kiểm định cặp): $n\approx\big((z_{1-\alpha/2}+z_{1-\beta})/d_z\big)^2$. Với $d_z=0.5$, $\alpha=0.05$, công suất $80\%$: $n\approx(1.96+0.84)^2/0.25\approx32$ người chơi (nên cộng hao hụt).
* **Dữ liệu thiếu và thoát giữa chừng** (informative censoring): báo cáo cách xử lý và phân tích độ nhạy.

### 7.5. Đạo đức nghiên cứu và pháp lý

Đồng ý có thông tin của người chơi; thông qua hội đồng đạo đức/đơn vị chủ quản; thông tin cho người tham gia sau thử nghiệm (debrief) nếu có điều chỉnh không báo trước; quyền rút khỏi nghiên cứu; xử lý người chơi dưới 18 tuổi; tuân thủ khung bảo vệ dữ liệu cá nhân hiện hành (mục 3.6). Cân nhắc rủi ro **thao túng**: hệ thống không được tối ưu để kéo dài thời gian chơi bằng cách khai thác điểm yếu tâm lý (xem Phần 9).

---

## PHẦN 8: TRIỂN KHAI TRONG GAME ENGINE (UNITY, ONNX)

### 8.1. Ràng buộc ONNX và Unity Sentis cần kiểm chứng

Mô hình được xuất bằng `export_tokenized_onnx` với đồ thị static-shape (`values` $[1,12,21]$, `mask` $[1,12,21]$, `context` $[1,C]$, `difficulty` $[1,10]$; đầu ra `logit`) và `opset_version=17` **[Đã triển khai v2]**. Theo tài liệu Unity (kiểm tra tháng 11/2026):

* Runtime suy luận hỗ trợ ONNX **opset 7–15**; mô hình ngoài khoảng này có thể nhập được nhưng kết quả không đảm bảo. Giá trị `opset_version=17` nằm ngoài khoảng.
* Toán tử **`GRU` nằm trong danh sách chưa được hỗ trợ**; `nn.GRU` khi xuất ONNX tạo đúng toán tử này.
* **Tên gọi:** Unity Sentis từng được đổi tên thành Inference Engine; về sau tên hiển thị trở lại Sentis nhưng gói phần mềm vẫn mang định danh `com.unity.ai.inference`. Báo cáo nên ghi *"Unity Sentis (Inference Engine), phiên bản X.Y"*.

**Phương án xử lý [Khuyến nghị]:**

1. Xuất với `opset_version` $\le15$ và đối chiếu danh sách toán tử được hỗ trợ của **đúng phiên bản package**.
2. Thay `nn.GRU` bằng bản **cài đặt thủ công trải phẳng (unrolled)** $L=12$ bước, chỉ dùng `Linear/MatMul`, `Sigmoid`, `Tanh`, `Mul`, `Add` (hoặc `nn.GRUCell` trong vòng lặp tĩnh), vì đồ thị chỉ gồm toán tử cơ bản.
3. Nếu vẫn không tương thích: thay khối chuỗi bằng TCN hoặc MLP trên đặc trưng tổng hợp và nêu rõ đánh đổi hiệu năng.
4. **Kiểm thử tương đương số học (parity test):** so sánh logit giữa PyTorch FP32, ONNX Runtime và Unity (FP32, FP16) trên hàng nghìn cửa sổ thật và ngẫu nhiên; báo cáo sai khác tối đa và trung bình (ví dụ mục tiêu $\le10^{-4}$ cho FP32; ngưỡng FP16 do bạn đặt trước) và sai khác ở *xác suất sau sigmoid* (ảnh hưởng quyết định). Chỉ tuyên bố "tương thích FP16" sau bước này.

**Các tuyên bố về hiệu năng là mục tiêu thiết kế, cần đo:** "Zero GC Alloc" phụ thuộc cách mã C# cấp phát tensor/mảng (cần bộ đệm cấp phát trước, `Dispose` đúng), phải kiểm bằng Unity Profiler; static shape chỉ là điều kiện cần.

### 8.2. Phân chia trách nhiệm giữa ONNX và mã C#

* **ONNX:** tokenizer, pooling, GRU, FiLM, head $\to$ `logit`.
* **C# (đều là phép tính nhẹ):** `CausalSessionNormalizer` (Welford, shrinkage), `EstimateGroupMaskUncertainty`, `EstimateStruggleProbability`, `OnlineTemperatureCalibrator` (MAP trên buffer logit), `SafeController`.
* **Độ chính xác số:** giữ normalizer, calibrator, controller ở FP32 (hoặc double cho $M_2$ của Welford khi phiên dài); chỉ mạng thần kinh mới cân nhắc FP16.
* **Chế độ suy luận:** gọi `model.eval()` trước khi xuất (Dropout trở thành đồng nhất).
* **Hiệu quả:** đồ thị hiện nhận $d$ là một đầu vào nên 3 ứng viên là 3 lượt chạy đầy đủ. **[Khuyến nghị]** tách thành *encoder* (`values`, `mask` $\to h_L$) và *head* (`h_L`, `context`, `difficulty` $\to$ `logit`), hoặc xuất đồ thị với batch tĩnh $B=3$; khi đó mã hóa lịch sử chỉ chạy một lần. Vì quyết định chỉ xảy ra tại checkpoint, ngân sách thời gian thoáng, nhưng vẫn nên đặt trần độ trễ (ms) trên thiết bị mục tiêu.

### 8.3. Chịu lỗi và triển khai an toàn **[Khuyến nghị]**

* **Dây chuyền dự phòng:** *WARMUP* (chưa đủ 12 snapshot) $\Rightarrow$ `HOLD`; đầu ra `NaN`, ngoại lệ hoặc quá thời gian $\Rightarrow$ `HOLD` (bất biến I7); thất bại liên tiếp $\Rightarrow$ *circuit breaker* chuyển sang rule-based hoặc độ khó cố định; công tắc tắt từ xa (kill switch).
* **Shadow mode:** chạy mô hình và ghi quyết định *mà không thực thi*; so sánh với nhánh đối chứng trước khi bật thật; phát hành dần theo tỷ lệ (canary).
* **Ghi log kiểm toán:** phiên bản mô hình, phiên bản contract, đầu vào đã chuẩn hóa, $\hat p$ từng ứng viên, $\mathcal{U}$, quyết định, *propensity* của chính sách (cho Phần 9).

### 8.4. Tái lập và quản trị mô hình

* Cố định hạt giống, phiên bản thư viện; lưu `split_manifest.json`, `metrics.json`, trọng số `.pt`, mô hình ONNX, băm của `canonical_feature_contract.json`.
* *Datasheet for Datasets* và *Model Card* mô tả nguồn dữ liệu, phạm vi sử dụng, giới hạn đã biết, kết quả theo nhóm.
* Bộ kiểm thử hiện có (`test_adaptive_components.py`, `test_pipeline.py`, `test_v2_hardening.py`) nên mở rộng với: test tương đương train–serve của normalizer (mục 2.6), test thuộc tính bất biến bộ điều khiển (mục 6.3), test khôi phục $\beta^\star$ của bộ hiệu chuẩn (mục 5.2), và test đẳng thức Confidence (mục 5.3).

---

## PHẦN 9: HƯỚNG PHÁT TRIỂN: BANDIT, ĐÁNH GIÁ NGOÀI CHÍNH SÁCH VÀ HỌC TĂNG CƯỜNG NGOẠI TUYẾN **[Mở rộng]**

### 9.1. Vị trí của hệ thống hiện tại

Hệ thống v2 là **chính sách tham lam một bước dựa trên mô hình** (predict-then-optimize): dự báo $\hat p(d)$ cho mỗi ứng viên rồi chọn theo $\text{Score}$. Các hướng dưới đây thay phần "chọn" bằng chính sách được học, nhưng chỉ có ý nghĩa khi có dữ liệu thích hợp.

### 9.2. Contextual Bandit và an toàn

* **Bài toán:** tại mỗi quyết định, thấy ngữ cảnh $x$, chọn hành động $a$ (độ khó), chỉ quan sát phần thưởng $r$ của hành động *đã chọn* (phản hồi một phần). Khác bài toán dự báo thuần ở chỗ phải *khám phá*.
* **Thuật toán:** LinUCB (cận tin cậy trên) và Thompson Sampling (lấy mẫu hậu nghiệm) cân bằng khai thác và khám phá; phiên bản **có ràng buộc an toàn** chỉ cho khám phá trong vùng an toàn (ví dụ chỉ trong các ứng viên có $\hat p\in[0.5,0.9]$).
* **Điều kiện tiên quyết:** log propensity $\mu(a\mid x)$ ngay từ bây giờ (mục 1.2), định nghĩa phần thưởng rõ ràng, và khám phá có kiểm soát trong biên an toàn.

### 9.3. Đánh giá ngoài chính sách (Off-Policy Evaluation)

Với dữ liệu log theo chính sách $\mu$, giá trị của chính sách mới $\pi$:
$$\hat V_{\text{IPS}}(\pi)=\frac1n\sum_{i}\frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}\,r_i,\qquad \hat V_{\text{DR}}(\pi)=\frac1n\sum_i\Big[\hat q(x_i,\pi)+\frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}\big(r_i-\hat q(x_i,a_i)\big)\Big],$$
với $\hat q(x,\pi)=\sum_a\pi(a\mid x)\hat q(x,a)$. IPS không chệch nếu propensity đúng và có positivity, nhưng phương sai lớn (cần clip hoặc chuẩn hóa tự thân, SNIPS); Doubly Robust không chệch nếu *hoặc* propensity *hoặc* mô hình $\hat q$ đúng. Chính mô hình $\hat p$ của v2 có thể đóng vai $\hat q$ (phương pháp "Direct Method").

### 9.4. Học tăng cường ngoại tuyến (Conservative Q-Learning)

* **MDP:** trạng thái $s$, hành động $a$, phần thưởng $r$, hệ số chiết khấu $\gamma$; mục tiêu $\max_\pi\mathbb E\big[\sum_t\gamma^tr_t\big]$ với Bellman $Q(s,a)=r+\gamma\,\mathbb E[\max_{a'}Q(s',a')]$.
* **Vấn đề ngoại tuyến:** truy vấn $Q$ tại hành động *ngoài dữ liệu* gây đánh giá quá cao (overestimation).
* **CQL** phạt giá trị $Q$ của hành động hiếm trong dữ liệu:
$$\min_Q\;\alpha\Big(\mathbb E_{s\sim\mathcal D}\big[\log\textstyle\sum_a e^{Q(s,a)}\big]-\mathbb E_{(s,a)\sim\mathcal D}\big[Q(s,a)\big]\Big)+\tfrac12\,\mathbb E\big[(Q-\mathcal B^\pi\hat Q)^2\big].$$
* **Điều kiện:** dữ liệu đủ lớn và bao phủ, định nghĩa phần thưởng (retention, độ dài phiên), và đánh giá bằng OPE trước khi thử online.

### 9.5. Đạo đức khi tối ưu hóa giữ chân người chơi

Tối ưu trực tiếp retention hoặc thời lượng phiên có nguy cơ **Goodhart** (chỉ số trở thành mục tiêu thì mất giá trị đo) và có thể dẫn tới khai thác điểm yếu tâm lý (mẫu thiết kế gây nghiện, dark pattern). Khuyến nghị: ràng buộc chính sách trong biên an toàn của mục 6.3, đưa chỉ số phúc lợi/trải nghiệm (PENS, GEQ) vào đánh giá, và nêu rõ giới hạn đạo đức trong báo cáo.

---

## PHẦN 10: PHỤ LỤC

### 10.1. Ma trận độ phủ khái niệm

| Khái niệm | Mục | Trạng thái |
|---|:---:|---|
| DDA, phân loại, định vị | 1.1 | Lý thuyết |
| Dự báo can thiệp, nhận dạng (A1–A3), S-learner | 1.2 | Lý thuyết + Khuyến nghị |
| Flow, Rasch/IRT, $P_{\text{target}}$ | 1.3 | Lý thuyết |
| Telemetry, Segment Contract, nhãn, kiểm duyệt | 1.4 | Đã triển khai + Khuyến nghị |
| Ba bậc khái quát hóa, zero/few-shot | 1.5 | Đã triển khai |
| 21 tín hiệu canonical (4/4/4/9) | 2.1 | Đã triển khai |
| Bất biến theo miền (đúng nghĩa) | 2.2 | Lý thuyết |
| Mặt nạ, MCAR/MAR/MNAR, không nội suy | 2.3 | Đã triển khai + Khuyến nghị |
| Sai phân, EMA | 2.4 | Đã triển khai |
| Log-ratio độ khó, vector 10 chiều, `empirical_prior` | 2.5 | Đã triển khai + Khuyến nghị |
| Welford, shrinkage, phương sai hỗn hợp, sàn $\sigma$ | 2.6 | Đã triển khai + Khuyến nghị |
| Rò rỉ, Group Split, tự tương quan | 3.1–3.2 | Đã triển khai |
| Confounding, Simpson, positivity | 3.3 | Lý thuyết + Khuyến nghị |
| Covariate/Label/Concept shift | 3.4 | Lý thuyết + Khuyến nghị |
| Căn chỉnh điểm quyết định | 3.5 | Khuyến nghị |
| Giả danh hóa HMAC, pháp lý | 3.6 | Khuyến nghị quản trị |
| Baseline (Majority, Logistic, GBDT) | 4.1 | Đã triển khai + Khuyến nghị |
| GRU | 4.2 | Đã triển khai |
| Tokenizer + Masked Gated Pooling | 4.3 | Đã triển khai |
| FiLM, độ nhạy theo $d$, đơn điệu | 4.4 | Đã triển khai + Khuyến nghị |
| BCE, Adam/AdamW, điều chuẩn | 4.5 | Đã triển khai + Cần xác minh |
| Mất cân bằng lớp | 4.6 | Khuyến nghị |
| Multi-task, ensemble | 4.7, 5.3 | Mở rộng |
| Brier, ECE, PR/ROC-AUC, Reliability Diagram | 5.1 | Đã triển khai + Khuyến nghị |
| Temperature Scaling, Batch-MAP, Platt | 5.2 | Đã triển khai + Khuyến nghị |
| Aleatoric/Epistemic, độ tin cậy theo độ phủ, Entropy | 5.3 | Đã triển khai + Khuyến nghị |
| Struggle Index | 5.4 | Đã triển khai + Khuyến nghị |
| Điều khiển vòng kín, Score, máy trạng thái, bất biến | 6.1–6.3 | Đã triển khai + Khuyến nghị |
| Dead band, debounce, lockout | 6.4 | Đã triển khai (đổi thuật ngữ) |
| Chỉ số bộ điều khiển, trải nghiệm | 6.5–6.6 | Lý thuyết + Khuyến nghị |
| Benchmark A/B và C–H, mô phỏng, RCT, thống kê, đạo đức | 7.1–7.5 | Đã triển khai (A, B) + Khuyến nghị |
| ONNX/Sentis, chịu lỗi, tái lập | 8.1–8.4 | Đã triển khai + Khuyến nghị |
| Bandit, OPE, CQL, đạo đức giữ chân | 9.1–9.5 | Mở rộng |

### 10.2. Bảng tham số hệ thống

| Tham số | Giá trị | Nguồn |
|---|---|---|
| Độ dài cửa sổ $L$; chu kỳ $\Delta$; bước trượt | 12; 5 s; 3 snapshot | Kiến trúc v2 |
| Số tín hiệu $F$; số chiều ngữ cảnh $C$; chiều độ khó | 21; 3; 10 (5 giá trị + 5 mask) | Kiến trúc v2 |
| Embedding / Token / GRU ẩn / FiLM | 32 / 64 / 64 / 64; head $112\to32\to1$ | Kiến trúc v2 |
| Tổng số tham số ($C=3$) | 36 194 | Tính từ cấu hình |
| Ngưỡng khuyết thời gian; đặc trưng khả dụng tối thiểu | 35%; 5 | Kiến trúc v2 |
| Chia tập theo người chơi | 70 / 15 / 15 | Kiến trúc v2 |
| $\sigma_{\min}$; clip chuẩn hóa | 0.05; $[-8,8]$ | Kiến trúc v2 |
| Trọng số nhóm; $u_{\text{base}}$; ngưỡng Confidence | $0.25$ mỗi nhóm; 0.05; 0.80 | Kiến trúc v2 |
| $P_{\text{target}}$; $\lambda_u$; $\lambda_c$ | 0.70; 0.25; 0.05 | Kiến trúc v2 |
| Trọng số struggle; ngưỡng struggle | 0.6 / 0.4; 0.75 | Kiến trúc v2 |
| Cooldown; streak; `progress` tối thiểu khi tăng | 60 s; 2; 0.30 | Kiến trúc v2 |
| Buffer cập nhật; cửa sổ trượt; số bước GD; $\tau$ | 20; 60; 25; $[0.7,2.0]$, $\tau_{\text{base}}=1.0$ | Kiến trúc v2 |
| $\kappa$ (`prior_weight`), $\lambda_{\text{prior}}$, $D_{\max}$, $r_{\max}$, tốc độ học, batch, dropout, weight decay, patience | *chưa nêu* | **[Cần xác minh]** |

### 10.3. Danh sách việc cần xác minh hoặc chỉnh trong mã nguồn

**Ưu tiên cao (ảnh hưởng tính đúng đắn của mô hình):**

1. Toán tử `GRU` và `opset 17` trên Unity Sentis (mục 8.1).
2. Dữ liệu huấn luyện có **biến thiên $\delta\ne0$** và ngẫu nhiên hóa độ khó không (mục 1.2, 3.3).
3. `empirical_prior`: tính từ người chơi train/out-of-fold? Có hàng mask $=0$ trong huấn luyện? (mục 2.5)
4. `masked_feature_dropout` có dùng trong *huấn luyện* không (mục 2.3).
5. Ablation chuẩn hóa và sàn $\sigma$ theo từng đặc trưng (mục 2.6).
6. Bộ hiệu chuẩn: thêm hệ số chặn/Platt, tối ưu theo $\beta$, giám sát $\operatorname{Var}(z)$ (mục 5.2).
7. $\mathcal{U}(d)$ không phụ thuộc $d$ (mục 5.3, 6.2).
8. Đánh giá theo điểm quyết định, `time_to_segment_start` (mục 3.5).
9. Quy tắc nhãn `quit` và crash-fallback (mục 1.4).
10. Kiểm tra độ nhạy theo $d$ và đơn điệu (mục 4.4).

**Ưu tiên trung bình (làm rõ đặc tả):** giá trị $\kappa$, $\lambda_{\text{prior}}$, $D_{\max}$, $r_{\max}$; ngưỡng "sẵn sàng tăng"; vai trò của $\text{Score}$; trạng thái streak khi bị `HOLD`; thứ tự trừ cooldown; $\hat p$ trong $P_{\text{struggle}}$ có hiệu chuẩn không; AdamW hay L2; vị trí Dropout; đơn vị `error_slope`; quy ước dấu $s_j$; `normalized_segment_index` khi game không có tổng phân đoạn; danh sách H1–H5.

**Ưu tiên thấp (thuật ngữ, tài liệu):** gọi đúng "Input-Coverage Confidence", "dead band + debounce + lockout", "giả danh hóa"; cập nhật căn cứ pháp lý; đồng bộ ma trận độ phủ và phụ lục.

### 10.4. Bảng thuật ngữ Anh–Việt

| Thuật ngữ | Giải thích ngắn |
|---|---|
| Aleatoric / Epistemic uncertainty | Bất định vốn có / bất định do thiếu hiểu biết của mô hình |
| Brier Score | Sai số bình phương trung bình của xác suất |
| Calibration (hiệu chuẩn) | Xác suất dự báo khớp tần suất thực |
| Causal normalization | Chuẩn hóa chỉ dùng thông tin đến thời điểm hiện tại |
| Concept drift / Covariate shift / Label shift | Đổi $P(Y\mid X)$ / $P(X)$ / $P(Y)$ |
| Confounding | Biến gây nhiễu ảnh hưởng đồng thời đến nguyên nhân và kết quả |
| Dead band | Vùng chết, không hành động |
| Debouncing | Xác nhận liên tiếp để lọc nhiễu xung |
| ECE | Sai số hiệu chuẩn kỳ vọng |
| FiLM | Điều chế đặc trưng theo tỷ lệ và dịch (scale + shift) |
| GRU | Đơn vị hồi quy có cổng |
| Group Split | Chia tập tách biệt theo nhóm (người chơi) |
| Ignorability | Hoán đổi được có điều kiện; điều kiện nhận dạng nhân quả |
| Lockout / Cooldown | Khóa thời gian sau mỗi lần đổi |
| Log-ratio | Logarit của tỉ số |
| MAP | Hậu nghiệm cực đại (Maximum A Posteriori) |
| Mask | Mặt nạ khả dụng của đặc trưng |
| Off-policy evaluation | Đánh giá chính sách mới bằng dữ liệu của chính sách khác |
| Positivity / Overlap | Mọi hành động có xác suất dương trong mỗi ngữ cảnh |
| Propensity | Xác suất chính sách chọn hành động |
| Proper scoring rule | Quy tắc chấm điểm cực tiểu tại xác suất thật |
| Pseudonymization | Giả danh hóa (còn liên kết được) |
| Rubber-banding | Cơ chế kéo cân bằng độ khó theo người chơi |
| S-learner | Một mô hình duy nhất nhận biến điều trị làm đầu vào |
| Safe checkpoint | Điểm an toàn để đổi độ khó |
| Segment | Phân đoạn game (wave, room, horizon cố định) |
| Shrinkage | Co cụm về giá trị tiên nghiệm |
| Temperature Scaling | Hiệu chuẩn bằng chia logit cho nhiệt độ |
| Welford | Thuật toán cập nhật trung bình/phương sai trực tuyến ổn định số |
| Zero-shot / Few-shot | Không dùng / dùng ít dữ liệu miền đích |

### 10.5. Tài liệu tham khảo gợi ý (cần đối chiếu nguồn gốc trước khi trích dẫn)

* Csikszentmihalyi, M. (1990). *Flow: The Psychology of Optimal Experience.*
* Sweetser, P., & Wyeth, P. (2005). GameFlow. *ACM Computers in Entertainment.*
* Hunicke, R. (2005). The case for dynamic difficulty adjustment in games. *ACE '05.*
* Zohaib, M. (2018). Dynamic Difficulty Adjustment (DDA) in computer games: a review. *Advances in Human-Computer Interaction.*
* Yannakakis, G. N., & Togelius, J. (2018). *Artificial Intelligence and Games.* Springer.
* Wilson, R. C., et al. (2019). The Eighty Five Percent Rule for optimal learning. *Nature Communications.*
* Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. *Technometrics.*
* Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder. *EMNLP.* Chung, J., et al. (2014). Empirical evaluation of gated recurrent neural networks.
* Perez, E., et al. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI.*
* Zaheer, M., et al. (2017). Deep Sets. *NeurIPS.* Gorishniy, Y., et al. (2021). Revisiting deep learning models for tabular data. *NeurIPS.*
* Kingma, D., & Ba, J. (2015). Adam. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization.
* Guo, C., et al. (2017). On calibration of modern neural networks. *ICML.* Platt, J. (1999). Probabilistic outputs for SVMs. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML.* Murphy, A. H. (1973). A new vector partition of the probability score. *J. Applied Meteorology.*
* Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS.* Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning? *NeurIPS.*
* Quiñonero-Candela, J., et al. (2009). *Dataset Shift in Machine Learning.* Saerens, M., et al. (2002). Adjusting the outputs of a classifier to new a priori probabilities. *Neural Computation.* Lipton, Z., et al. (2018). Detecting and correcting for label shift. *ICML.* Ben-David, S., et al. (2010). A theory of learning from different domains. *Machine Learning.*
* Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. *JMLR.* Arjovsky, M., et al. (2019). Invariant risk minimization.
* Hernán, M., & Robins, J. (2020). *Causal Inference: What If.* Künzel, S., et al. (2019). Metalearners for estimating heterogeneous treatment effects. *PNAS.* Little, R., & Rubin, D. (2019). *Statistical Analysis with Missing Data.*
* Li, L., et al. (2010). A contextual-bandit approach to personalized news article recommendation. *WWW.* Dudík, M., et al. (2011). Doubly robust policy evaluation and learning. *ICML.* Kumar, A., et al. (2020). Conservative Q-learning for offline reinforcement learning. *NeurIPS.*
* Alshiekh, M., et al. (2018). Safe reinforcement learning via shielding. *AAAI.*
* Che, Z., et al. (2018). Recurrent neural networks for multivariate time series with missing values. *Scientific Reports.*
* Ryan, R., Rigby, C., & Przybylski, A. (2006). The motivational pull of video games (PENS). IJsselsteijn, W., et al. (2013). Game Experience Questionnaire. Jackson, S., & Eklund, R. (2002). Flow State Scale-2. Hart, S., & Staveland, L. (1988). NASA-TLX.
* Gebru, T., et al. (2021). Datasheets for datasets. *CACM.* Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\*.*
* Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15; Nghị định 356/2025/NĐ-CP. Tài liệu Unity Sentis/Inference Engine: danh sách toán tử ONNX được hỗ trợ (đối chiếu đúng phiên bản package).


---

## PHỤ LỤC AUDIT V3 — ĐỐI CHIẾU SAU KHI KHẮC PHỤC

Phụ lục này **ghi đè trạng thái triển khai** của các đoạn mô tả cũ khi chúng mâu thuẫn với mã nguồn hiện tại. Kết quả rà soát được thực hiện trên branch `main` sau khi đồng bộ repository.

### A. Các vấn đề đã khắc phục

1. **Đánh giá mô hình:** `src/ddamodels/train.py` hiện báo cáo Brier decomposition gần đúng theo Murphy (`reliability`, `resolution`, `uncertainty`), ECE toàn cục, ECE vùng vận hành `[0.6, 0.8]`, ROC-AUC, PR-AUC success và PR-AUC failure. Khi báo cáo bài nghiên cứu, dùng `pr_auc_failure` làm chỉ số chính nếu failure là lớp dương.
2. **Hiệu chuẩn:** `OnlineTemperatureCalibrator` vẫn giữ tên tương thích nhưng đã chuyển thành affine logit scaling: `logit(p_cal) = beta * logit(p_raw) + bias`. `beta` có prior về 1, `bias` có prior về 0, cả hai đều bị giới hạn; `temperature = 1/beta` chỉ là biểu diễn tương thích. Cơ chế vẫn cập nhật theo sliding window và đủ batch, không cập nhật từng segment.
3. **FiLM:** khối điều chế dùng `h'=(1+delta_gamma(d))*h+beta(d)` và khởi tạo `delta_gamma`, `beta` bằng 0. Do đó tại thời điểm khởi tạo, FiLM là ánh xạ đồng nhất thay vì làm triệt tiêu hidden state.
4. **Regularization:** Tokenized FiLM GRU có dropout ở hidden/head, `feature_group_dropout` trong train và `empirical_prior_dropout`. Có validation loss, lưu best state và early stopping theo patience.
5. **ONNX/Sentis:** backbone dùng GRU tĩnh tự triển khai bằng `Linear/Sigmoid/Tanh/Mul/Add`, không phụ thuộc toán tử ONNX `GRU`; export dùng `opset_version=15`. Tương thích Unity vẫn phải xác nhận bằng parity test trên đúng package Sentis/Inference Engine và thiết bị mục tiêu.
6. **Controller:** dữ liệu lỗi/NaN bị giữ nguyên difficulty; cooldown được trừ bằng `dt_seconds` trước khi kiểm tra; candidate bị giới hạn trong phạm vi ±1 và `D_max` nếu được truyền. Score dùng để xác định hướng ứng viên tốt nhất, sau đó struggle/increase gate và streak xác nhận hành động.
7. **Chuẩn hóa:** causal normalizer dùng floor theo từng feature `max(raw_scale, min_scale_ratio * game_scale, min_scale)`, tránh dùng một sàn tuyệt đối cho mọi đơn vị đo. `causal_normalize_canonical_frame()` thực hiện cùng thứ tự `transform(current) -> update(current)` với runtime.
8. **Dữ liệu v2:** `build_tokenized_window_rows()` và `causal_normalize_canonical_frame()` đã có trong `data.py`; mask được tạo từ khả dụng thật, còn difficulty mask được giữ riêng rồi ghép thành vector 10 chiều khi load.

### B. Những gì đã kiểm chứng tự động

Bộ test hiện tại chạy thành công: **20 test, 0 failure, 1 skip** trong môi trường không cài PyTorch. Test bao phủ batch calibration, affine bias, mask semantics, causal train/serve order, builder v2, group split, controller cooldown/bounds và các invariant hardening. Test forward/export mạng neural cần chạy lại trong môi trường có PyTorch + ONNX Runtime; không được coi là đã kiểm chứng chỉ vì Python compile thành công.

### C. Giới hạn còn lại — không được tuyên bố quá mức

* Input-Coverage Confidence vẫn không phải epistemic uncertainty. Nó không phát hiện OOD khi toàn bộ feature có mặt; ensemble hoặc MC-Dropout là phần mở rộng cần benchmark riêng.
* `Score` hiện có uncertainty theo candidate nếu caller cung cấp uncertainty khác nhau. Nếu uncertainty chỉ là mask coverage dùng chung, hạng tử uncertainty triệt tiêu khi so sánh ứng viên và chỉ còn tác dụng ở cổng confidence.
* Mô hình hiện vẫn là S-learner dự báo xác suất thành công có điều kiện theo candidate. Không được diễn giải thành xác suất can thiệp `do(d)` nếu chưa có randomized difficulty logging, positivity và kiểm soát confounding.
* Affine calibration sửa được cả scale và bias nhưng không biến dữ liệu on-policy thành dữ liệu có positivity. Cần log propensity và đánh giá Leave-Session-Out/few-shot trước khi kết luận khái quát hóa.
* Việc export được ONNX không đồng nghĩa Unity Sentis chạy được. Bắt buộc kiểm tra numerical parity PyTorch–ONNX Runtime–Unity trên FP32 và FP16, đồng thời đo latency/GC bằng Unity Profiler.
* Do môi trường hiện tại không có PyTorch, chưa thể báo cáo metric thực tế của Tokenized FiLM GRU hoặc xác nhận graph ONNX. Đây là bước nghiệm thu bắt buộc trước khi chạy user study.

### D. Quy trình nghiệm thu tiếp theo

1. Cài dependency đầy đủ, train trên dataset có ít nhất ba nhóm player/session độc lập và có validation không rỗng.
2. Chạy `scripts/train_adaptive.py`, kiểm tra `metrics.json`, `tokenized_film_gru.pt`, `tokenized_film_gru.onnx` và split manifest.
3. Dùng ONNX Runtime để parity-test tối thiểu 1.000 window; sau đó lặp lại trong Unity Sentis đúng phiên bản package.
4. Chạy synthetic calibration test với bias biết trước, group-dropout ablation, candidate sensitivity/monotonicity và closed-loop simulator.
5. Chỉ sau các bước trên mới mở shadow mode rồi canary mode trong game; mọi lỗi inference phải giữ difficulty hoặc chuyển rule-based fallback.
