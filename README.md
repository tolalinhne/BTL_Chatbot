# 📚 FIT.Subject Chatbot

## 📖 Giới Thiệu
Dự án **FIT.Subject Chatbot** là một chatbot học thuật được xây dựng để hỗ trợ sinh viên và giảng viên trong việc giải đáp các thắc mắc liên quan đến môn **Lý Thuyết Thông Tin** (Information Theory). Chatbot sử dụng mô hình ngôn ngữ Gemini của Google và tích hợp tìm kiếm ngữ nghĩa (semantic search) để cung cấp câu trả lời chính xác, đầy đủ và có ngữ cảnh. Dự án được phát triển với mục tiêu hỗ trợ học thuật, đồng thời thân thiện và dễ sử dụng.

## ✨ Tính Năng Chính
- **Hỗ trợ học thuật**: Trả lời các câu hỏi liên quan đến Lý Thuyết Thông Tin với độ chính xác cao, dựa trên dữ liệu đã được nhúng (embedding).
- **Tìm kiếm ngữ nghĩa**: Sử dụng mô hình `paraphrase-multilingual-MiniLM-L12-v2` để tìm kiếm ngữ cảnh phù hợp với câu hỏi.
- **Quản lý cuộc trò chuyện**: Người dùng có thể tạo, xóa, đổi tên và chuyển đổi giữa các cuộc trò chuyện.
- **Lịch sử trò chuyện**: Lưu trữ và hiển thị lịch sử trò chuyện, tự động dọn dẹp sau 30 phút không hoạt động.
- **Định dạng trả lời**: Sử dụng Markdown để làm nổi bật nội dung quan trọng (in đậm, in nghiêng, danh sách, bảng).
- **Giao diện thân thiện**: Giao diện web đơn giản, dễ sử dụng, hỗ trợ trên nhiều thiết bị.

## 🛠️ Công Nghệ Sử Dụng
- **Ngôn ngữ lập trình**: Python
- **Framework**: Flask (web framework)
- **Mô hình AI**: Google Gemini (`gemini-2.0-flash`)
- **Tìm kiếm ngữ nghĩa**: Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Frontend**: HTML, CSS, JavaScript (giao diện tĩnh với template Jinja2)
- **Logging**: Python `logging` để ghi lại hoạt động của chatbot
- **Khác**: Markdown, Bleach (làm sạch HTML), NumPy

## ⚙️ Cài Đặt
### Yêu Cầu Hệ Thống
- Hệ điều hành: Windows, macOS, hoặc Linux
- Python: >= 3.8
- Dung lượng trống: Khoảng 1GB (bao gồm các thư viện và mô hình)

### Hướng Dẫn Cài Đặt
1. **Clone repository về máy**:
   ```
   git clone https://github.com/tolalinhne/BTL_Chatbot.git
   ```
2. **Di chuyển vào thư mục dự án**:
   ```
   cd BTL_Chatbot
   ```
3. **Tạo môi trường ảo (khuyến nghị)**:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
4. **Cài đặt các thư viện cần thiết**:
   ```
   pip install -r requirements.txt
   ```
   > **Lưu ý**: File `requirements.txt` chứa các thư viện cần thiết như `flask`, `google-generativeai`, `sentence-transformers`, `markdown`, `bleach`, `numpy`, `gunicorn`. Ví dụ nội dung file:
   ```
   flask==2.0.1
   google-generativeai==0.3.0
   sentence-transformers==2.2.2
   markdown==3.4.1
   bleach==5.0.1
   numpy==1.23.5
   gunicorn==20.1.0
   ```
5. **Cấu hình API Key cho Gemini**:
   - Trong file `app.py`, thay thế giá trị `my_api_key_gemini` bằng API Key của bạn:
     ```python
     my_api_key_gemini = "YOUR_GEMINI_API_KEY"
     ```
6. **Chuẩn bị dữ liệu nhúng (embedding)**:
   - Nếu bạn có dữ liệu nhúng sẵn, đặt file `embedding_data.pkl` vào thư mục gốc của dự án.
   - Nếu không, bạn có thể tạo dữ liệu nhúng bằng cách chạy script `build_data_embedding.py`:
     ```
     python build_data_embedding.py
     ```
   - Đảm bảo thư mục `DATA/` chứa các file văn bản cần nhúng (nếu có).
7. **Chạy ứng dụng**:
   - Sử dụng Flask để chạy trong môi trường phát triển:
     ```
     python app.py
     ```
   - Hoặc sử dụng Gunicorn cho môi trường production (cấu hình trong `gunicorn_config.py`):
     ```
     gunicorn -c gunicorn_config.py app:app
     ```
   Mặc định, ứng dụng chạy trên `http://127.0.0.1:5000`.

## 🚀 Sử Dụng
- **Truy cập ứng dụng**: Mở trình duyệt và truy cập `http://127.0.0.1:5000`.
- **Đặt câu hỏi**: Nhập câu hỏi liên quan đến Lý Thuyết Thông Tin vào ô nhập liệu và nhấn "Ask".
- **Quản lý cuộc trò chuyện**:
  - Nhấn vào biểu tượng menu (☰) để xem danh sách các cuộc trò chuyện.
  - Nhấn "+ New Conversation" để tạo cuộc trò chuyện mới.
  - Nhấn "Rename Conversation" để đổi tên cuộc trò chuyện hiện tại.
  - Nhấn "Delete" để xóa một cuộc trò chuyện.
- **Xóa lịch sử**: Nhấn "Clear History" để xóa lịch sử trò chuyện trong phiên hiện tại.

## 🤝 Đóng Góp
Chúng tôi hoan nghênh mọi đóng góp! Vui lòng làm theo các bước sau:
1. Fork repository.
2. Tạo một branch mới (`git checkout -b feature/[ten-tinh-nang]`).
3. Commit thay đổi (`git commit -m "Mô tả thay đổi"`).
4. Push lên branch (`git push origin feature/[ten-tinh-nang]`).
5. Tạo Pull Request.

## 📜 Giấy Phép
Dự án này được phân phối dưới **MIT License**. Xem file `LICENSE` để biết thêm chi tiết.

## 📬 Liên Hệ
- Email: [email@example.com]
- GitHub: [tolalinhne](https://github.com/tolalinhne)