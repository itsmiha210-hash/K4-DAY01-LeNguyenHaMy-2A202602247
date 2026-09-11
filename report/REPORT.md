# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** GPU (Tesla T4)

**Python / PyTorch / Ultralytics:** Python 3.10 / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `K4-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
    - `class_id`: 468
    - `class_name`: "cab"
    - `rank`: 1
    - `score`: 0.510915
    - `taxonomy_name`: "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
    - Record này gán một nhãn duy nhất ở cấp độ toàn ảnh là `cab` với độ tin cậy ~51.09% (rank 1).
    - Đây là bài toán phân loại đơn nhãn, mô hình nhìn nhận toàn bộ ảnh như một thực thể chung chứ không định vị vị trí không gian.
    - Dù ảnh `traffic` là một cảnh giao thông phức tạp với rất nhiều chủ thể cùng xuất hiện, mô hình phân loại vẫn buộc phải chọn ra một nhãn duy nhất có điểm số cao nhất từ taxonomy ImageNet-1K, dẫn đến việc bỏ sót toàn bộ các chủ thể còn lại trong ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
    - Class list được huấn luyện bởi nhóm phát triển tập dữ liệu ImageNet-1K với 1000 lớp.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
    - `class_id`: Đảm bảo định danh duy nhất cho máy tính xử lý và lập chỉ mục trong cơ sở dữ liệu, tránh lỗi do xử lý chuỗi ký tự hoặc sai lệch ngôn ngữ.
    - `class_name`: Giúp annotator, reviewer, kỹ sư đọc hiểu trực quan hơn.
    - `taxonomy_name`: Xác định không gian nhãn và ngữ cảnh phân loại (ví dụ: `ImageNet-1K` khác với `COCO` hay `OpenImages`). Cùng một tên lớp hoặc mã ID ở hai taxonomy khác nhau có thể mang định nghĩa và phạm vi ngữ nghĩa khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
    - Guideline cần xác định quy tắc ưu tiên rõ ràng cho bài toán đơn nhãn: ví dụ ưu tiên chủ thể chiếm diện tích pixel lớn nhất ở tiền cảnh, hoặc chủ thể ở vị trí trung tâm, hoặc theo mục đích cụ thể.
    - Quy định khi nào cần chuyển sang bài toán đa nhãn nếu một nhãn đơn lẻ không thể đại diện cho toàn bộ nội dung bức ảnh.
    - Đưa ra quy tắc xử lý khi cảnh quá đông đúc/mơ hồ (ví dụ: gắn nhãn bao quát như `traffic_scene` hoặc gắn cờ escalation gửi cấp quản lý phê duyệt).
- Vì sao model score không phải ground truth?
    - Model score (confidence score) chỉ là xác suất thống kê đầu ra của mô hình (sau hàm softmax), phản ánh độ tự tin nội tại của mạng nơ-ron dựa trên dữ liệu đã học. Score cao không đồng nghĩa với việc kết quả đó đúng trong thực tế (mô hình vẫn có thể tự tin dự đoán sai).
    - Ground truth là sự thật khách quan đã được annotator hoặc reviewer kiểm định, xác nhận độc lập dựa theo taxonomy và guideline chuẩn của dự án.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
    - `class_name`: "person"
    - `score`: 0.912625
    - `bbox_xyxy`: [385.33, 69.24, 498.92, 348.92]
    - `bbox_width`: 113.58
    - `bbox_height`: 279.68
    - (Ghi chú: gốc tọa độ (0, 0) ở góc trên bên trái, đơn vị tính bằng pixel).
- Diễn giải vị trí box bằng lời:
    - Hộp bao quanh người (`person`) nằm ở nửa bên phải bức ảnh. Tọa độ ngang bắt đầu từ x ≈ 385.33 px đến x ≈ 498.92 px (chiều rộng ~113.58 px); tọa độ dọc bắt đầu từ đỉnh đầu y ≈ 69.24 px kéo dài xuống chân y ≈ 348.92 px (chiều cao ~279.68 px). Box bao trọn người phụ nữ đang đứng cạnh bàn bếp.
- So sánh số prediction ở hai threshold:
    - Tại ngưỡng `DETECTION_SCORE_THRESHOLD = 0.35`: mô hình trả về 11 dự đoán (gồm 2 person, 5 bowl, 2 oven, 2 cup).
    - Tại ngưỡng cao hơn `threshold = 0.60`: số dự đoán giảm xuống còn 6 (gồm 2 person, 2 bowl, 2 oven; các vật thể nhỏ/mờ như cup và các bowl điểm thấp bị loại bỏ).
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
    - Độ bao phủ: Khi hạ ngưỡng threshold (từ 0.60 xuống 0.35), độ bao phủ tăng lên, giúp phát hiện thêm các vật thể nhỏ/khó nhận dạng, giảm nguy cơ bỏ sót vật thể thực tế. Tuy nhiên nếu đặt threshold quá thấp sẽ kéo theo nhiều dự đoán sai (false positive).
    - Khối lượng reviewer cần xem: Hạ threshold làm tăng số lượng hộp dự đoán, reviewer phải tốn nhiều thời gian lọc và xóa bỏ các hộp rác. Ngược lại, nâng threshold cao giúp reviewer xem ít hộp hơn nhưng phải tự tay vẽ thêm các đối tượng bị mô hình bỏ sót (false negative).
- Đề xuất một quy tắc box chặt:
    - Cả 4 cạnh của hộp (x_min, y_min, x_max, y_max) phải ôm sát tối đa mép nhìn thấy được ngoài cùng của vật thể, khoảng cách hở tới viền vật thể không vượt quá 2-3 pixel và tuyệt đối không cắt xén bất kỳ bộ phận nhìn thấy nào của vật thể.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
    - Trong trường hợp bị che khuất: Guideline cần quy định chỉ vẽ hộp bao quanh phần nhìn thấy được (visible box) hay ước lượng cả phần bị che khuất (amodal box); đặt ngưỡng phần trăm diện tích tối thiểu được nhìn thấy để gán nhãn (ví dụ ≥ 15%); gắn nhãn thuộc tính `occluded`.
    - Trong trường hợp bị cắt mép: Quy định rõ vật thể nằm ở rìa ảnh bị cắt mép có được tính không và gắn nhãn thuộc tính `truncated`.
    - Khi vật thể bị che khuất quá nặng hoặc mờ đến mức con người không thể chắc chắn bằng mắt thường, annotator gắn cờ  để reviewer hoặc lead dự án quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
    - `instance_id`: "kitchen-001"
    - `class_name`: "person"
    - `score`: 0.899318
    - Số điểm: 348 điểm
    - Một phần `polygon_xy`: `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 71.0], [442.0, 72.0], [442.0, 73.0], [441.0, 74.0], ...]`
- Polygon bổ sung chi tiết gì so với box?
    - Bounding box chỉ là một hình chữ nhật thô bao quanh và chứa nhiều điểm ảnh nền dư thừa. Polygon cung cấp tọa độ viền ôm sát hình dáng thực tế của đối tượng, phân định chính xác từng điểm ảnh thuộc về vật thể hay nền.
    - Nắm bắt được các đường cong, chỗ lõm, tư thế giơ tay, bước chân, và giải quyết triệt để tình trạng hai vật thể nằm cạnh/đè lên nhau mà bounding box bị chồng lấn khó phân biệt.
- `instance_id` dùng để làm gì và không phải loại ID nào?
    - Dùng để phân biệt từng cá thể riêng lẻ trong cùng một bức ảnh của bài thực hành (ví dụ: `kitchen-001` và `kitchen-009` cùng thuộc lớp `person` nhưng là hai cá thể độc lập).
    - Không phải là `class_id` (mã lớp định danh chung cho danh mục đối tượng), `tracking_id` (ID theo dõi vật thể xuyên suốt các khung hình video theo thời gian), hay ID toàn cục vĩnh viễn trong cơ sở dữ liệu.
- Đề xuất một quy tắc biên mask:
    - Đường viền polygon phải bám sát ranh giới phân cách giữa vật thể và hậu cảnh (dung sai sai số cho phép không quá 1-2 pixel), đảm bảo các đường biên cong mượt mà, không bị răng cưa giả tạo và không bao gồm các điểm ảnh nền xung quanh.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
    - Quy định rõ ranh giới phân chia (chia đôi viền, không để khe hở khoảng trống nhưng cũng không được để hai mask chồng lấn lên nhau nếu cùng ở một mặt phẳng).
    - Quy định lấy biên theo đường viền lõi nhìn rõ nhất (core edge) hay lấy theo ranh giới gradient độ sáng trung bình.
    - Chỉ gán nhãn và tạo mask cho phần nhìn thấy thực tế (visible mask). Nếu mức độ mờ/che khuất khiến ranh giới hoàn toàn không xác định được, annotator phải gắn cờ escalation gửi reviewer.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn danh mục cấp ảnh (`class_id`, `class_name` theo ImageNet-1K) | Ảnh có nhiều chủ thể đồng thời (xe buýt, taxi, người, minibus), không rõ chủ thể trọng tâm | Áp dụng đúng quy tắc ưu tiên trong guideline (chủ thể trung tâm/diện tích lớn); không đoán theo cảm tính | Kiểm tra nhãn có tuân thủ đúng quy tắc ưu tiên không; rà soát các trường hợp ảnh ngoại lệ |
| Phát hiện vật thể | Tập hợp các tọa độ hộp chữ nhật `[x_min, y_min, x_max, y_max]` kèm lớp | Mô hình bỏ sót các vật thể nhỏ/mờ (cốc chén, thìa); hộp quá rộng chứa nhiều nền hoặc nhầm lẫn giữa tủ bếp và lò nướng | Vẽ lại hộp ôm khít mép nhìn thấy; bổ sung thêm các vật thể bị bỏ sót; điều chỉnh đúng nhãn lớp | So sánh độ khít của box (IoU); kiểm tra vật thể bị sót (false negative) và vật thể gán nhãn sai (false positive) |
| Instance segmentation | Tập hợp đa giác tọa độ đỉnh `[[x1,y1], [x2,y2], ...]` cho từng cá thể | Đa giác lấn viền vào nền; hai vật thể tiếp xúc bị dính chung đường biên; bỏ sót chi tiết mảnh (cán thìa) | Chỉnh sửa các điểm neo ôm sát viền thực tế; tách riêng từng cá thể độc lập không để dính viền | Phóng to kiểm tra độ chính xác của đường viền; đảm bảo không có khoảng hở và không chồng lấn vô lý |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
    - Tuyệt đối không tải lên, chia sẻ hoặc đưa dữ liệu nhạy cảm/dữ liệu nhận dạng cá nhân (PII như họ tên, MSSV, thông tin liên lạc cá nhân, khuôn mặt riêng tư, biển số xe) hoặc dữ liệu nội bộ của tổ chức vào mã nguồn công khai, Colab hay repository public. Chỉ sử dụng đúng các bộ dữ liệu mẫu công khai được cấp phép (COCO, ImageNet).
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
    - Dừng thao tác xử lý và thông báo ngay cho người có thẩm quyền cao hơn để cách ly dữ liệu và xử lý theo quy trình.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
