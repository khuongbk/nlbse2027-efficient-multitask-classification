# Teacher, student and exits / Teacher, student và exit

[English](#english) · [Tiếng Việt](#tiếng-việt)

Numbers refer to the submission `A2_R11_all_full` unless stated otherwise. Code: `nlbse27ee/model.py`
(architecture), `nlbse27ee/train.py` (training), `nlbse27ee/exits.py` (thresholds, margins, FLOPs
simulator).

```
                      input: <s> <encoder-only> </s> [<satd_c>|<satd_w>] compacted code </s>   (<= 512 tokens)
                                               |
                                      embeddings (word + position, LayerNorm)
                                               |-----------> exit 0  (attention pool -> Linear(768, 2))
                                     layer 1  (teacher layer 2)
                                               |-----------> exit 1
                                     layer 2  (teacher layer 4)
                                               |-----------> exit 2
                                              ...
                                     layer 6  (teacher layer 12)
                                               '-----------> exit 6  (final: every remaining row stops here)

  At each exit k, a row leaves when |z[k,label] - tau[k,label]| >= m[k,label] for both labels.
  Rows that leave are removed from the batch; the rest is trimmed to its longest remaining row.
```

---

## English

### 1. The teacher

**What it is.** `microsoft/unixcoder-base-nine`, a 12-layer RoBERTa-style encoder (hidden size 768,
12 attention heads, feed-forward size 3,072, 125.3M parameters), fine-tuned on the competition data.
Its head is masked mean pooling over the tokens followed by `Linear(768, 2)`, one logit per label
(Vulnerable, MAT), trained with binary cross-entropy.

**Input.** The same input the student sees: whitespace compacted, an optional flag token
(`<satd_c>` when `TODO`/`FIXME`/`HACK`/`XXX` occurs inside a comment, `<satd_w>` when it occurs
elsewhere), UniXcoder's encoder-only format, at most 512 tokens.

**Training.** lr 3e-5 (head 3e-4), effective batch 64 (16 per GPU x 4 GPUs, DDP), 10-epoch linear
schedule with warmup, best epoch 8 on the selection split (final teacher: all rows, stopped at epoch 8).

**Result.** Mean F1 0.9425 on the public test, the most accurate model we trained, but 834,264 GFLOPs
and about 62 s on a T4, so its competition score is only 0.8974. It is never submitted.

**Its two jobs.**
1. **Source of weights.** The student is cut out of it (section 2), so the student starts from a model
   that already understands code and the task.
2. **Source of soft targets.** Its logits tell the student how confident a strong model is about each
   row. Here a problem appeared: on its own training rows the teacher's logits have the sign of the
   true label on 99.94% (Vulnerable) and 99.99% (MAT) of the rows, i.e. it has memorized them and its
   soft targets carry almost nothing beyond the labels. Hence the **fold teachers**: the 197,419
   research-stage training rows are split into 4 stratified folds; fold teacher *f* is trained without
   fold *f* and scores it. Every training row therefore gets a soft target from a teacher that never
   saw it (out-of-fold, OOF). These disagree with the label on 3.37% of the rows, so they carry
   real information ("this row looks like the other class", "this row is ambiguous").

### 2. The student

**How it is built** (`build_from_teacher`). No new network is trained from scratch; the student is
copied out of the fine-tuned teacher:

| Part | Teacher | Student |
|---|---|---|
| Embeddings | 768-dim, full vocabulary | identical copy (plus the two flag tokens) |
| Transformer layers | 12 | 6: teacher layers 2, 4, 6, 8, 10, 12 (indices 1, 3, ..., 11 counting from 0) |
| Attention per layer | 12 heads, 768 | identical copy |
| Feed-forward per layer | 3,072 neurons | the 1,536 most important neurons of that layer |
| Heads | 1 classifier | 7 exit heads, each starting from the teacher's classifier |
| Parameters | 125.3M | 68.7M |

Taking every second layer keeps the teacher's full depth range (low-level to high-level features)
at half the cost.

**Feed-forward pruning.** In each layer, a neuron *j*'s importance is
`mean over real tokens of |GELU(fc1(x))_j|` (how strongly it fires, measured on 4,096 training rows)
times `||fc2[:, j]||` (how much its output can change the layer's output). The 1,536 highest-scoring
of 3,072 neurons are kept and the rest are deleted, together with their rows of `fc1` and columns of
`fc2`. Under the profiler, a student layer costs about 2/3 of a teacher layer per token (attention
projections unchanged, feed-forward halved).

**Why 68.7M parameters but very little compute.** About 40M of the parameters are the embedding
table, which is a lookup and costs no FLOPs. The compute is in the 6 layers (about 4.7M parameters
each), and early exit means most rows run only one of them.

**Training.** All 7 exits are trained together. For exit *k* with logits `z_k`:

```
loss_k = 0.5 * BCE(z_k, y) + 0.5 * T^2 * BCE(z_k / T, sigmoid(t / T)),   T = 2
loss   = sum_k w_k * loss_k,   w_k proportional to k + 1 (exit 0: 1/28, ..., exit 6: 7/28)
```

`y` is the true label and `t` the out-of-fold teacher logit. Deeper exits get more weight because they
handle the hard rows; shallow exits still get enough signal to become confident on easy rows.
Optimizer AdamW, lr 4e-5 (heads 4e-4), effective batch 128, 14-epoch schedule (final model stopped at
epoch 13, the epoch selected in the research stage), fp16, and an **EMA** of the weights (decay 0.999)
that is what gets evaluated and exported.

### 3. The exits

**Exit heads.** Each exit is an **attention-pooling head**: a `Linear(768, 1)` gives every token a
score, a softmax over the real tokens turns the scores into weights, the weighted sum of token vectors
goes through `Linear(768, 2)`. The score layer is initialized to zero, so every head starts as exact
mean pooling (and exactly as the teacher's head) and learns during training which tokens to look at,
for example the flag token or calls such as `memcpy`.

**Exit 0** reads the embeddings, before any transformer layer. On its own it reaches mean F1 0.84,
which is enough to settle about a quarter of the functions for the cost of a lookup and one small head.

**The decision rule.** At exit *k* the model has logits `z[k, Vulnerable]` and `z[k, MAT]`, plus two
calibrated numbers per label:

- a **threshold** `tau[k, label]`: the label is positive when `z > tau`. It is the F1-optimal threshold
  of that exit on the calibration rows;
- a **margin** `m[k, label]`: how far from the threshold the logit must be to trust it.

A row **leaves** at exit *k* when `|z[k, label] - tau[k, label]| >= m[k, label]` for **both** labels,
and its labels are `z[k, label] > tau[k, label]`. Otherwise it continues to the next layer. Exit 6 is
final: every row still present stops there. A margin of 0 means "always confident about this label";
a margin of infinity means "never stop here".

**Worked example** (illustrative logits, real thresholds and margins of exit 0:
`tau = (-0.882, 0.788)`, `m = (7.0, 2.0)`). A short function with `// TODO: handle errors` in a comment:

| Label | `z` at exit 0 | distance `|z - tau|` | margin | confident? | label |
|---|---:|---:|---:|---|---|
| Vulnerable | -8.5 | 7.6 | 7.0 | yes | 0 |
| MAT | 6.0 | 5.2 | 2.0 | yes | 1 |

Both labels are confident, so the function leaves at exit 0 without running any transformer layer.
If the Vulnerable logit had been -4.0 (distance 3.1 < 7.0), the function would have continued to
layer 1 and been decided there or later.

**Why the measured GFLOPs really drop.** Inside `forward`, rows that leave are removed from the batch
tensor and the remaining batch is cut to the length of its longest remaining row. The next layer only
sees the rows and tokens that are left, so the profiler counts fewer FLOPs; nothing is simulated or
masked. Combined with sorting rows by length before batching, short easy functions leave early and long
batches shrink as they go.

**How thresholds and margins are chosen.** Thresholds: per exit and label, independently. Margins:
chosen to **maximize the official competition score**, not F1. The search needs the GFLOPs and runtime
of each candidate policy, so `FlopsModel` reproduces the profiler's count from the row lengths and
exit decisions (within 0.02% of the profiler) and runtime is taken proportional to GFLOPs. Coordinate
ascent then goes over the non-final exits; at each exit both labels' margins are searched jointly over
`{0, 0.5, ..., 4, 5, 6, 7, 8, 10, 12, inf}^2`, up to 3 passes.

**Where the rows of the public test leave** (submission):

| Exit | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean F1 if every row stopped here | 0.8448 | 0.9196 | 0.9336 | 0.9390 | 0.9408 | 0.9405 | 0.9415 |
| Rows that leave here | 7,423 | 15,894 | 5,660 | 1,156 | 486 | 28 | 8 |

76% of the functions are decided before the second layer; a row runs 1.07 of the 6 layers on average.
The early-exit model's mean F1 (0.9392) is close to running all 6 layers on every row (0.9415) at a
fraction of the compute.

---

## Tiếng Việt

### 1. Teacher

**Là gì.** `microsoft/unixcoder-base-nine`, một encoder kiểu RoBERTa 12 lớp (chiều ẩn 768, 12 attention
head, FFN 3.072, 125,3M tham số), fine-tune trên dữ liệu thi. Đầu phân loại: lấy trung bình các token
thật (mean pooling) rồi qua `Linear(768, 2)`, ra một logit cho mỗi nhãn (Vulnerable, MAT), huấn luyện
bằng binary cross-entropy.

**Input.** Giống hệt input của student: đã gộp khoảng trắng, có thể có token cờ (`<satd_c>` khi
`TODO`/`FIXME`/`HACK`/`XXX` nằm trong comment, `<satd_w>` khi nằm ở chỗ khác), định dạng encoder-only của
UniXcoder, tối đa 512 token.

**Huấn luyện.** lr 3e-5 (head 3e-4), batch hiệu dụng 64 (16 mỗi GPU × 4 GPU, DDP), lịch 10 epoch có
warmup, epoch tốt nhất là 8 trên tập chọn (teacher cuối: toàn bộ dữ liệu, dừng ở epoch 8).

**Kết quả.** Mean F1 0,9425 trên public test, chính xác nhất trong mọi model đã train, nhưng 834.264
GFLOPs và khoảng 62 s trên T4, nên điểm thi chỉ 0,8974. Teacher không bao giờ được nộp.

**Hai vai trò.**
1. **Nguồn trọng số.** Student được "cắt" ra từ teacher (phần 2), nên khởi đầu đã hiểu code và hiểu bài.
2. **Nguồn soft label.** Logit của teacher cho student biết một model mạnh tự tin đến đâu với từng hàng.
   Vấn đề: trên chính các hàng nó đã học, logit của teacher cùng dấu với nhãn thật ở 99,94% (Vulnerable)
   và 99,99% (MAT) — nó đã thuộc lòng, nên soft label gần như chỉ là nhãn cứng. Vì vậy có **teacher theo
   fold**: 197.419 hàng train của giai đoạn nghiên cứu được chia thành 4 fold stratified; teacher fold
   *f* được train khi bỏ fold *f* ra, rồi chấm điểm fold đó. Mỗi hàng nhận soft label từ một teacher
   **chưa từng thấy nó** (out-of-fold, OOF). Các soft label này khác dấu với nhãn ở 3,37% hàng, tức là
   mang thông tin thật ("hàng này trông giống lớp kia", "hàng này mơ hồ").

### 2. Student

**Cách tạo** (`build_from_teacher`). Không train mạng mới từ đầu; student được sao chép ra từ teacher
đã fine-tune:

| Thành phần | Teacher | Student |
|---|---|---|
| Embedding | 768 chiều, đủ từ vựng | sao chép nguyên (thêm 2 token cờ) |
| Lớp transformer | 12 | 6: lớp 2, 4, 6, 8, 10, 12 của teacher (chỉ số 1, 3, …, 11 nếu đếm từ 0) |
| Attention mỗi lớp | 12 head, 768 | sao chép nguyên |
| FFN mỗi lớp | 3.072 neuron | 1.536 neuron quan trọng nhất của lớp đó |
| Đầu phân loại | 1 | 7 exit, mỗi exit khởi tạo từ classifier của teacher |
| Tham số | 125,3M | 68,7M |

Lấy cách một lớp giữ được toàn bộ dải độ sâu của teacher (từ đặc trưng cấp thấp tới cấp cao) với một
nửa chi phí.

**Cắt FFN.** Trong mỗi lớp, độ quan trọng của neuron *j* là
`trung bình trên các token thật của |GELU(fc1(x))_j|` (neuron kích hoạt mạnh đến đâu, đo trên 4.096 hàng
train) nhân với `||fc2[:, j]||` (đầu ra của neuron ảnh hưởng tới đầu ra của lớp đến đâu). Giữ 1.536 trong
3.072 neuron điểm cao nhất, xoá phần còn lại cùng hàng tương ứng của `fc1` và cột tương ứng của `fc2`.
Theo cách profiler đếm, mỗi lớp student tốn khoảng 2/3 một lớp teacher trên mỗi token (phần chiếu của
attention giữ nguyên, FFN giảm một nửa).

**Vì sao 68,7M tham số mà compute rất ít.** Khoảng 40M tham số là bảng embedding — chỉ là tra bảng,
không tốn FLOPs. Compute nằm ở 6 lớp (mỗi lớp khoảng 4,7M tham số), và nhờ early exit, phần lớn hàng chỉ
chạy một lớp.

**Huấn luyện.** Cả 7 exit được train cùng lúc. Với exit *k* có logit `z_k`:

```
loss_k = 0.5 * BCE(z_k, y) + 0.5 * T^2 * BCE(z_k / T, sigmoid(t / T)),   T = 2
loss   = sum_k w_k * loss_k,   w_k tỉ lệ với k + 1 (exit 0: 1/28, ..., exit 6: 7/28)
```

`y` là nhãn thật, `t` là logit OOF của teacher. Exit sâu có trọng số lớn hơn vì chúng xử lý các hàng
khó; exit nông vẫn đủ tín hiệu để tự tin trên các hàng dễ. AdamW, lr 4e-5 (head 4e-4), batch hiệu dụng
128, lịch 14 epoch (model cuối dừng ở epoch 13, epoch đã chọn ở giai đoạn nghiên cứu), fp16, và **EMA**
của trọng số (decay 0,999) — bản EMA là bản được đánh giá và export.

### 3. Exit

**Đầu exit.** Mỗi exit là một **đầu attention pooling**: `Linear(768, 1)` cho mỗi token một điểm, softmax
trên các token thật biến điểm thành trọng số, tổng có trọng số của các vector token đi qua
`Linear(768, 2)`. Lớp tính điểm khởi tạo bằng 0, nên ban đầu mỗi exit đúng là mean pooling (và đúng bằng
đầu của teacher), rồi trong lúc train học được nên nhìn vào token nào, ví dụ token cờ hay lời gọi
`memcpy`.

**Exit 0** đọc ngay embedding, trước mọi lớp transformer. Một mình nó đạt mean F1 0,84 — đủ để quyết
định khoảng một phần tư số hàm chỉ với chi phí tra bảng và một đầu nhỏ.

**Quy tắc quyết định.** Ở exit *k*, model có logit `z[k, Vulnerable]` và `z[k, MAT]`, cùng hai con số
đã calibrate cho mỗi nhãn:

- **ngưỡng** `tau[k, nhãn]`: nhãn dương khi `z > tau`; là ngưỡng tối ưu F1 của exit đó trên tập calibrate;
- **margin** `m[k, nhãn]`: logit phải cách ngưỡng bao xa thì mới tin được.

Một hàng **rời** ở exit *k* khi `|z[k, nhãn] - tau[k, nhãn]| >= m[k, nhãn]` với **cả hai** nhãn, và nhãn
của nó là `z[k, nhãn] > tau[k, nhãn]`. Nếu không, nó đi tiếp lớp sau. Exit 6 là exit cuối: hàng nào còn
lại đều dừng ở đó. Margin 0 nghĩa là "luôn tin nhãn này"; margin vô cùng nghĩa là "không bao giờ dừng ở
đây".

**Ví dụ** (logit minh hoạ, ngưỡng và margin thật của exit 0: `tau = (-0,882; 0,788)`,
`m = (7,0; 2,0)`). Một hàm ngắn có `// TODO: handle errors` trong comment:

| Nhãn | `z` ở exit 0 | khoảng cách `|z - tau|` | margin | tin được? | nhãn |
|---|---:|---:|---:|---|---|
| Vulnerable | -8,5 | 7,6 | 7,0 | có | 0 |
| MAT | 6,0 | 5,2 | 2,0 | có | 1 |

Cả hai nhãn đều tin được, nên hàm rời ở exit 0 mà không chạy lớp transformer nào. Nếu logit Vulnerable là
-4,0 (khoảng cách 3,1 < 7,0) thì hàm sẽ đi tiếp lớp 1 và được quyết định ở đó hoặc sau đó.

**Vì sao GFLOPs đo được thật sự giảm.** Trong `forward`, các hàng đã rời bị xoá khỏi tensor của batch, và
batch còn lại được cắt về độ dài của hàng dài nhất còn lại. Lớp sau chỉ thấy các hàng và token còn lại,
nên profiler đếm ít FLOPs hơn — không mô phỏng, không che mask. Kết hợp với sắp xếp hàng theo độ dài
trước khi chia batch, các hàm ngắn và dễ rời sớm, còn các batch dài thì co lại dần.

**Chọn ngưỡng và margin thế nào.** Ngưỡng: mỗi exit, mỗi nhãn, độc lập. Margin: chọn để **tối đa điểm thi
chính thức**, không phải F1. Việc tìm cần GFLOPs và runtime của từng chính sách thử, nên `FlopsModel`
tính lại số FLOPs của profiler từ độ dài các hàng và quyết định exit (sai số dưới 0,02%), còn runtime
lấy tỉ lệ với GFLOPs. Sau đó coordinate ascent đi qua các exit không phải cuối; ở mỗi exit, margin của
cả hai nhãn được tìm đồng thời trên `{0; 0,5; …; 4; 5; 6; 7; 8; 10; 12; ∞}²`, tối đa 3 lượt.

**Các hàm trong public test rời ở đâu** (model nộp):

| Exit | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean F1 nếu mọi hàng dừng ở đây | 0,8448 | 0,9196 | 0,9336 | 0,9390 | 0,9408 | 0,9405 | 0,9415 |
| Số hàng rời ở đây | 7.423 | 15.894 | 5.660 | 1.156 | 486 | 28 | 8 |

76% số hàm được quyết định trước lớp thứ hai; trung bình mỗi hàng chạy 1,07 trong 6 lớp. Mean F1 của
model có early exit (0,9392) gần bằng chạy đủ 6 lớp cho mọi hàng (0,9415) mà chỉ tốn một phần nhỏ compute.
