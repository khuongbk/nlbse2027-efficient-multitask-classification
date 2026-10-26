# Câu chuyện của lời giải: từ CodeBERT 0,73 điểm đến early-exit UniXcoder ~0,958 điểm

Tài liệu này kể lại lời giải theo đúng trình tự nó được xây dựng: nhìn ra điều gì trong đề bài, thử gì,
cái gì có tác dụng, cái gì không, và vì sao model cuối cùng trông như vậy. Số liệu kỹ thuật đầy đủ nằm
trong [README](../README.md); quy tắc thi và nơi đáp ứng nằm trong [COMPLIANCE](COMPLIANCE.md).

---

## 1. Đọc kỹ công thức chấm điểm: compute đáng giá ngang accuracy

Đề bài: một model đọc hàm C/C++ (`Function`) và dự đoán hai nhãn nhị phân, **Vulnerable** và **MAT**
(nợ kỹ thuật tự thừa nhận). Điểm:

```
score = 0.6 * mean F1 + 0.2 * max((1000 - runtime_s)/1000, 0) + 0.2 * max((3e6 - GFLOPs)/3e6, 0)
```

Baseline CodeBERT của ban tổ chức: F1 0,9102, 192 s, 2.666.510 GFLOPs, **điểm 0,7299**.

Điều quan trọng nhất của cả bài nằm ở đây. Thử tách điểm baseline ra:

| Thành phần | Baseline | Tối đa | Baseline đang mất |
|---|---:|---:|---:|
| 0,6 × F1 | 0,546 | 0,600 | 0,054 |
| 0,2 × runtime | 0,162 | 0,200 | 0,038 |
| 0,2 × GFLOPs | 0,022 | 0,200 | **0,178** |

Mức trần GFLOPs (3 triệu) chỉ cao hơn baseline một chút, nên baseline gần như mất trắng 0,2 điểm của
phần GFLOPs. Dù có tăng F1 lên tuyệt đối 1,0 thì cũng chỉ thêm được 0,054 điểm; trong khi giảm GFLOPs
xuống gần 0 thì thêm được gần 0,18. **Bài này thực chất là bài toán "làm sao tốn ít compute nhất cho
mỗi input mà vẫn giữ được F1".** Toàn bộ thiết kế phía sau xoay quanh câu đó.

Một chi tiết thứ hai: GFLOPs được đo bằng `torch.profiler(with_flops=True)` trên 10 lần forward mỗi
batch 600 hàng, fp16 autocast. Profiler đếm phép nhân ma trận (addmm/bmm) và phép elementwise, nên
những gì làm giảm số token thật sự đi qua các lớp Linear đều được ghi nhận trực tiếp vào điểm.

## 2. Chọn nền: UniXcoder thay vì CodeBERT

Cả hai đều là encoder 12 lớp, 768 chiều, cùng kích thước. UniXcoder (`microsoft/unixcoder-base-nine`)
được pretrain trên code nhiều ngôn ngữ hơn, có cả C/C++, và có chế độ encoder-only riêng
(`<s> <encoder-only> </s> code </s>`). Teacher 12 lớp fine-tune trên dữ liệu thi đạt F1 0,9428 trên
public test (đo held-out), cao hơn baseline CodeBERT 0,9102 khá nhiều. Nhưng chính teacher đó chỉ được
**0,8975 điểm**: F1 tốt, nhưng 834k GFLOPs. Điều này xác nhận nhận định ở phần 1: model "giỏi" nhưng
nặng thì thua.

Vậy nhiệm vụ là: **giữ được độ chính xác của teacher trong một model rẻ hơn hàng chục lần.**

## 3. Trước khi tối ưu: một quy trình không rò rỉ dữ liệu

Ở vòng thử nghiệm đầu, model được chọn trên chính public test. Một thành viên trong nhóm chỉ ra vấn đề:
nếu public test vừa dùng để chọn model, vừa dùng để báo cáo, thì con số báo cáo bị lạc quan. Thực tế
xác nhận điều đó: teacher vòng đầu đạt 0,9455 trên public khi được chọn trên public, nhưng chỉ 0,9428
khi public được giữ riêng.

Nên toàn bộ công việc được làm lại theo quy trình sạch:

| Dữ liệu | Dùng để |
|---|---|
| 92% train (197.419 hàng) | huấn luyện mọi teacher và student trong giai đoạn nghiên cứu |
| 8% train (17.167 hàng, stratified theo cả hai nhãn, seed 42) | chọn epoch, ngưỡng, margin early exit, và chọn cấu hình thắng |
| public test (30.655 hàng) | đo **một lần** cho mỗi model, sau khi mọi lựa chọn đã xong |
| 100% train (214.586 hàng) | huấn luyện lại công thức thắng để nộp |

Quy tắc được cố định từ đầu: **chọn bằng điểm trên tập 8%, không nhìn public test để chọn.** Phần 8
giải thích vì sao quy tắc này được giữ ngay cả khi public test "gợi ý" một lựa chọn khác.

Ở bước cuối, ngưỡng của model nộp được calibrate trên public test, vì hướng dẫn của ban tổ chức nói rõ
split này dành cho "model tuning, selection, and validation reporting". Nhưng con số không thiên lệch
để báo cáo trong paper là con số của cùng công thức ở giai đoạn nghiên cứu, khi public test hoàn toàn
được giữ riêng.

## 4. Giai đoạn A: tìm kiến trúc (13 run)

### 4.1. Bắt đầu: một student nhỏ hơn teacher

Cách trực tiếp nhất để giảm compute là bớt lớp. Student đầu tiên (**R0**) lấy 4 trong 12 lớp của teacher
(lớp 2, 5, 8, 11), khởi tạo từ trọng số teacher và distill từ logit của teacher. R0 đạt F1 0,9297,
324k GFLOPs, **0,9312 điểm** — đã hơn baseline 0,20 điểm chỉ nhờ nhỏ hơn.

### 4.2. Input rẻ hơn và giàu thông tin hơn (R1)

Hai quan sát về dữ liệu:

- **Code có rất nhiều khoảng trắng** (thụt lề, dòng trống). Với tokenizer, mỗi cụm khoảng trắng là token
  tốn compute mà không mang nghĩa. Gộp khoảng trắng giảm **16% số token**, và số hàng bị cắt ở 512
  token giảm từ 3,8% xuống 0,5% — tức là model còn thấy được nhiều code hơn.
- **MAT gần như được định nghĩa bởi tag** `TODO`/`FIXME`/`HACK`/`XXX`. Khi tag nằm trong comment, 99,7%
  là MAT. Nhưng nếu hàm dài và bị cắt ở 512 token thì tag có thể bị mất. Nên cờ được tính trên toàn bộ
  văn bản gốc rồi chèn thành một token đặc biệt ở đầu input: `<satd_c>` (tag trong comment) hoặc
  `<satd_w>` (tag ở chỗ khác). Đây hoàn toàn là một hàm của `Function`, không dùng thông tin ngoài.

Kết quả R1: F1 +0,0045 (MAT +0,012), GFLOPs −14%, **0,9377 điểm**.

### 4.3. Ý tưởng trung tâm: early exit

Không phải hàm nào cũng khó như nhau. Một hàm có `// TODO` trong comment và trông "sạch" thì không cần
chạy hết 4 hay 6 lớp mới biết nhãn. Early exit gắn một đầu phân loại sau mỗi lớp; khi model đã đủ chắc
chắn ở một lớp, hàng đó **rời batch** và không đi tiếp.

Cụ thể:

- Mỗi exit *k* có ngưỡng riêng cho từng nhãn `tau[k, nhãn]` (ngưỡng tối ưu F1 trên tập chọn).
- Một hàng rời ở exit *k* khi logit của **cả hai nhãn** cách ngưỡng ít nhất một margin
  `m[k, nhãn]`, tức là model chắc chắn về cả hai.
- Khi hàng rời, nó bị loại khỏi tensor; batch còn lại được cắt về độ dài của hàng dài nhất còn lại. Nhờ
  vậy profiler thật sự thấy ít FLOPs hơn — không phải mô phỏng.
- Margin không được chọn bằng tay mà **được tìm để tối đa trực tiếp công thức điểm chính thức**. Để
  làm được điều đó, cần dự đoán GFLOPs của mỗi tổ hợp margin mà không phải chạy profiler; một bộ mô
  phỏng FLOPs được viết lại theo đúng cách profiler đếm, sai số dưới 0,02%. Việc tìm là coordinate
  ascent, ở mỗi exit tìm đồng thời margin của cả hai nhãn trên lưới
  {0; 0,5; …; 4; 5; 6; 7; 8; 10; 12; ∞}². (Phiên bản đầu tìm từng nhãn riêng thì bị kẹt ở (∞, ∞) — không
  hàng nào được thoát — vì đổi một nhãn đơn lẻ không bao giờ đủ để một hàng rời đi; tìm đồng thời
  sửa được lỗi đó.)

Early exit (R2) **giảm một nửa GFLOPs** của cùng một mạng: 278k → 137k.

### 4.4. Sâu hơn nhưng hẹp hơn (R4–R11)

Khi đã có early exit, câu hỏi đổi khác: thêm lớp không còn tốn compute cho mọi hàng, chỉ cho những hàng
khó. Nên các biến thể thử độ sâu và độ rộng:

| Run | Thay đổi | F1 public (held-out) | GFLOPs | Điểm |
|---|---|---:|---:|---:|
| R4 | 3 lớp | 0,9182 | 130k | 0,9398 |
| R2 | 4 lớp | 0,9205 | 137k | 0,9404 |
| R7 | 6 lớp (1, 3, 5, 7, 9, 11) | 0,9364 | 162k | 0,9476 |
| R9 | 6 lớp, FFN 2.048 | 0,9356 | 125k | 0,9499 |
| R8 | 6 lớp, FFN 1.536 | 0,9368 | 111k | 0,9517 |
| **R11** | R8, lịch 14 epoch | 0,9348 | 111k | 0,9503 |

- **6 lớp thắng 4 và 3 lớp** khi có early exit: các hàng dễ vẫn rời sớm, các hàng khó có đủ độ sâu.
- **Cắt FFN** (lớp feed-forward chiếm khoảng 2/3 FLOPs mỗi lớp): giữ 1.536 trong 3.072 neuron quan
  trọng nhất, đo bằng mức kích hoạt trung bình |GELU| nhân với độ lớn cột tương ứng của lớp chiếu ra.
  F1 gần như giữ nguyên, compute giảm khoảng một phần ba.

**R11** có điểm cao nhất trên tập chọn (0,9557) nên thắng giai đoạn A. Lưu ý: trên public, R8 nhỉnh hơn
R11 — nhưng theo quy tắc, public không được dùng để chọn.

### 4.5. Một phát hiện bất ngờ: distillation không giúp gì (R3)

R3 là R2 nhưng bỏ distillation, chỉ học từ nhãn. Kết quả: **R3 bằng hoặc hơn R2**. Lý do được tìm ra
khi soi logit của teacher: trên chính các hàng nó được huấn luyện, logit của teacher **cùng dấu với
nhãn thật ở 99,94% hàng** (Vulnerable) và 99,99% (MAT). Teacher đã thuộc lòng tập train, nên "soft label"
của nó không khác gì nhãn cứng — student không học thêm được gì. Phát hiện này dẫn trực tiếp tới một
trong các cải tiến ở giai đoạn A2.

## 5. Giai đoạn A2: ba cải tiến trên R11

Mọi biến thể A2 đều xây trên R11 (người thắng giai đoạn A), để kết quả là một mạch liên tục chứ không
phải các nhánh rời rạc.

### 5.1. Exit 0 và attention pooling

- **Exit 0**: thêm một đầu phân loại ngay trên embedding, **trước mọi lớp transformer**. Nghe có vẻ vô
  lý, nhưng exit 0 một mình đạt mean F1 0,84 — đủ để quyết định chắc chắn rất nhiều hàng mà không tốn
  một lớp nào. Embedding đã có thông tin từ vựng (token `<satd_c>`, các API nguy hiểm…).
- **Attention pooling**: thay trung bình đều các token bằng một trung bình có trọng số học được. Lớp tính
  trọng số được khởi tạo bằng 0, nên ban đầu nó đúng là mean pooling và không làm hỏng điểm xuất phát;
  sau đó nó học được nên "nhìn" vào token nào (ví dụ token cờ, lời gọi `memcpy`). Mỗi exit nông nhờ vậy
  mạnh hơn, quyết định được nhiều hàng hơn.

### 5.2. EMA của trọng số

Giữ trung bình trượt (decay 0,999) của trọng số trong lúc train, và dùng trọng số đó để đánh giá và nộp.
Rẻ, ổn định, thêm khoảng +0,001 trên tập chọn.

### 5.3. Distillation out-of-fold

Từ phát hiện ở 4.5: muốn soft label có ích thì teacher không được nhìn thấy hàng đó lúc train. Tập
train 92% được chia thành 4 fold stratified; 4 teacher được train, mỗi teacher bỏ ra một fold và chấm
điểm fold đó. Như vậy mỗi hàng nhận soft label từ một teacher **chưa từng thấy nó**. Các soft label này
khác dấu với nhãn ở 3,37% hàng (đồng ý 96,63%, so với 99,94% của teacher thuộc lòng) — tức là chúng
thật sự mang thông tin ngoài nhãn, như "hàng này khó/mơ hồ".

### 5.4. Kết quả

| Biến thể | Điểm tập chọn | F1 public (held-out) | GFLOPs | Điểm public (ước tính) |
|---|---:|---:|---:|---:|
| R11 | 0,9557 | 0,9348 | 111.121 | 0,9503 |
| + EMA | 0,9567 | 0,9356 | 115.143 | 0,9506 |
| + EMA + OOF | 0,9551 | 0,9367 | 101.284 | 0,9524 |
| + EMA + exit 0 + attention | 0,9588 | 0,9361 | 63.689 | 0,9548 |
| **+ cả ba (`A2_R11_all`)** | **0,9604** | 0,9334 | **53.214** | 0,9541 |

- **Exit 0 + attention là bước tiến lớn nhất**: GFLOPs từ 111k xuống 54–64k.
- EMA giúp một chút; OOF tăng F1 khi đứng riêng (+0,0019) nhưng không nhất quán khi kết hợp.
- `A2_R11_all` thắng trên tập chọn.

## 6. Vì sao chọn `all` mà không chọn `ema_exit0_attn`?

Trên public test, `ema_exit0_attn` nhỉnh hơn `all` 0,0007 điểm. Có thể hỏi: sao không chọn nó?

1. **Quy tắc đã cố định trước**: chọn theo tập chọn, ở đó `all` thắng (0,9604 so với 0,9588). Nếu
   nhìn public rồi đổi lựa chọn, public test thành tập chọn model và con số báo cáo không còn sạch — đúng
   loại rò rỉ mà quy trình được thiết kế để tránh.
2. **0,0007 nằm trong nhiễu.** Hai model hơn nhau ở hai mặt khác nhau: `ema_exit0_attn` F1 cao hơn
   (0,9361 so với 0,9334), `all` rẻ hơn (53k so với 64k GFLOPs). Đổi seed hay epoch có thể đảo thứ tự.
3. **Không mất gì trên thực tế**: sau khi train lại trên toàn bộ dữ liệu, `all` đạt ~0,9575, cao hơn cả
   hai con số trên.

Trong paper, cả hai được báo cáo như ablation, và nêu rõ tiêu chí chọn.

## 7. Giai đoạn B: model nộp

Công thức thắng (`A2_R11_all`) và teacher được train lại trên **toàn bộ 214.586 hàng**, dừng ở đúng
epoch đã chọn ở giai đoạn nghiên cứu (teacher 8, student 13), rồi calibrate ngưỡng và margin trên public
test.

| Model | Vuln F1 | MAT F1 | Mean F1 | GFLOPs | Runtime | Điểm |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CodeBERT (T4 chính thức) | 0,9390 | 0,8814 | 0,9102 | 2.666.510 | 192,1 s | 0,7299 |
| Teacher UniXcoder 12 lớp | 0,9508 | 0,9342 | 0,9425 | 834.264 | ~62 s | 0,8974 |
| R11 full | 0,9417 | 0,9347 | 0,9382 | 111.055 | ~14,9 s | ~0,9525 |
| **Model nộp `A2_R11_all_full`** | **0,9436** | **0,9349** | **0,9392** | **55.443** | **~11,6 s** | **~0,9575** |

Runtime và điểm hiện là ước tính từ GPU L40S (nhân hệ số 5,32, lấy từ tỉ lệ thời gian của chính baseline
trên L40S và T4); sẽ thay bằng số đo trên Colab T4. F1 và GFLOPs không phụ thuộc GPU.

Model nộp có 68,7M tham số (teacher 125,3M). **Trung bình mỗi hàng chỉ chạy 1,07 trong 6 lớp**:

| Exit | 0 (embedding) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Số hàng rời ở đây | 7.423 | 15.894 | 5.660 | 1.156 | 486 | 28 | 8 |

24% hàng được quyết định chỉ bằng embedding, thêm 52% sau lớp đầu tiên. Chỉ 36 trên 30.655 hàng đi tới
hai lớp cuối.

## 8. Chi tiết nhỏ, tác động lớn: padding

Notebook baseline tokenize toàn bộ tập test một lần với `padding=True`, nên mọi hàng bị pad tới cùng một
độ dài (512). Đề không yêu cầu cách padding này — nó chỉ là cấu hình của baseline.

Một điều không hiển nhiên: **pad theo từng batch (dynamic) một mình không giúp gì** ở batch 600, vì gần
như batch 600 hàng nào theo thứ tự gốc cũng có ít nhất một hàng 512 token. Phải **sắp xếp hàng theo độ
dài trước**, rồi mới pad theo batch; dự đoán được ghi lại đúng thứ tự gốc. Số đo (không giả định):

| Padding | Token đã pad | F1 | GFLOPs |
|---|---:|---:|---:|
| global (như baseline) | 15.695.360 | 0,9392 | 156.277 |
| dynamic, thứ tự gốc | 15.588.005 | 0,9392 | 156.277 |
| **sắp xếp + dynamic** | **4.910.360** | 0,9392 | **55.443** |

Trong mức giảm **48 lần** GFLOPs so với baseline, **model đóng góp 17 lần** (đo dưới đúng cách padding
của baseline), **sắp xếp + pad theo batch đóng góp 2,8 lần**. F1 không đổi. Cách phân tách này quan trọng
cho paper: đóng góp của phương pháp không bị lẫn với mẹo đánh giá.

## 9. Model sai ở đâu: phân tích MAT

MAT rất mất cân bằng: 505 trên 30.655 hàm (1,65%). Model nộp: 445 TP, 2 FP, 60 FN (precision 0,9955,
recall 0,8812). Chia theo cờ:

| Nhóm | Số hàm | MAT thật | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| tag trong comment | 444 | 443 | 443 | 1 | 0 |
| tag ở chỗ khác | 7 | 1 | 1 | 1 | 0 |
| không có tag | 30.204 | 61 | 1 | 0 | 60 |

**Cả 60 lỗi bỏ sót đều là hàm không có tag nào.** Trên mọi hàm có tag, model không bỏ sót ca nào. 61
trên 505 hàm MAT (12,1%) không có dấu hiệu gì trong `Function` — recall của MAT bị giới hạn bởi dữ liệu,
không phải bởi model.

## 10. Bài học rút ra

1. **Đọc công thức chấm trước khi chọn model.** Ở đây compute đáng giá hơn F1; mọi quyết định đều theo
   đó.
2. **Tối ưu thẳng vào hàm mục tiêu.** Margin early exit được tìm trên chính công thức điểm, với bộ mô
   phỏng FLOPs khớp profiler.
3. **Phần lớn input là dễ.** Ba phần tư số hàm được quyết định trước khi qua lớp thứ hai.
4. **Distillation cần teacher không thuộc lòng.** Soft label của teacher trên chính tập train của nó
   gần như là nhãn cứng; out-of-fold mới mang thông tin.
5. **Đo, đừng giả định.** Dynamic padding "đương nhiên giúp" — nhưng đo thì không, trừ khi có sắp xếp.
6. **Giữ public test sạch.** Mọi model đều kém hơn trên public so với tập chọn khoảng 0,005–0,008 F1;
   chọn trên public sẽ làm con số báo cáo lạc quan đúng chừng đó.

## 11. Hạn chế và việc còn lại

- Runtime và điểm của model nộp là ước tính cho tới khi chạy notebook Colab T4
  (`notebooks/NLBSE27_Submission_Colab_T4.ipynb`).
- Mỗi cấu hình mới chạy một seed; các chênh lệch dưới ~0,002 điểm không nên diễn giải là có ý nghĩa.
- Hiệu quả của OOF distillation chưa nhất quán; cần thêm seed để kết luận.
- Model nộp calibrate ngưỡng trên public test (được hướng dẫn cho phép); con số không thiên lệch của
  công thức là `A2_R11_all` (F1 0,9334, ~0,9541 điểm).
