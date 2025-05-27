from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import markdown
import re
import os
import logging
from datetime import datetime
import json
import uuid
import numpy as np
from learning_system import LearningSystem

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chatbot")

# Load model đa ngôn ngữ cho embedding
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

import os
my_api_key_gemini = "AIzaSyCxHoogDhMpOob5fGfH-nCNabO0EiWPJs4"

genai.configure(api_key=my_api_key_gemini)
gemini_model = genai.GenerativeModel('models/gemini-2.0-flash')

app = Flask(__name__)
# Thiết lập secret key cho session
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key_for_session")

# Định nghĩa cấu trúc lưu trữ cho lịch sử trò chuyện
chat_history = {}
# Định nghĩa cấu trúc lưu trữ cho danh sách các cuộc trò chuyện
user_conversations = {}
# Thời gian lưu trữ lịch sử chat tối đa (phút)
CHAT_HISTORY_EXPIRATION = 30

# Load embedding data khi khởi động app
EMBEDDING_DATA_PATH = 'embedding_data.pkl'
embedding_data = None
if os.path.exists(EMBEDDING_DATA_PATH):
    import pickle
    with open(EMBEDDING_DATA_PATH, 'rb') as f:
        embedding_data = pickle.load(f)
    embedding_chunks = embedding_data['chunks']
    embedding_sources = embedding_data['sources']
    embedding_vectors = embedding_data['embeddings']
else:
    embedding_chunks = []
    embedding_sources = []
    embedding_vectors = None

# Khởi tạo hệ thống học hỏi
learning_system = LearningSystem(embedding_model)

# Define your 404 error handler to redirect to the index page
@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))

def chunk_text_files(filepaths, chunk_size=3, overlap=1):
    """Chia các file thành đoạn văn bản có ngữ cảnh hoàn chỉnh với overlap giữa các chunk"""
    all_chunks = []
    chunk_sources = []
    
    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Tách thành các câu - regex cải tiến để xử lý nhiều trường hợp dấu câu
            sentences = re.split(r'(?<=[.!?])\s+', content)
            sentences = [s for s in sentences if s.strip()]
            
            # Tạo các chunk với overlap
            for i in range(0, len(sentences), chunk_size - overlap):
                if i + chunk_size <= len(sentences):
                    chunk = " ".join(sentences[i:i + chunk_size])
                    all_chunks.append(chunk)
                    chunk_sources.append(filepath)
                elif i < len(sentences):  # Xử lý chunk cuối cùng nếu không đủ kích thước
                    chunk = " ".join(sentences[i:])
                    all_chunks.append(chunk)
                    chunk_sources.append(filepath)
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {filepath}: {str(e)}")
            continue
            
    return all_chunks, chunk_sources

def semantic_search_context(question, filepaths=None, top_k=5):
    """Tìm kiếm ngữ cảnh liên quan đến câu hỏi dùng embedding đã lưu"""
    if embedding_vectors is None or not embedding_chunks:
        return "", []
    # Tính embedding cho câu hỏi
    question_embedding = embedding_model.encode(question, convert_to_numpy=True)
    # Tính cosine similarity
    scores = np.dot(embedding_vectors, question_embedding) / (
        np.linalg.norm(embedding_vectors, axis=1) * np.linalg.norm(question_embedding) + 1e-8)
    # Lấy top_k đoạn liên quan nhất
    top_indices = np.argsort(scores)[-top_k:][::-1]
    context_chunks = []
    sources_used = []
    for idx in top_indices:
        score = scores[idx]
        if score > 0.3:
            context_chunks.append(embedding_chunks[idx])
            src = embedding_sources[idx]
            if src not in sources_used:
                sources_used.append(src)
    context = " ".join(context_chunks)
    return context, sources_used

def create_augmented_prompt(question, context, sources=None, chat_history_text=None):
    """Tạo prompt tối ưu với few-shot example, chain-of-thought và lịch sử trò chuyện"""
    # Nếu có lịch sử trò chuyện, đưa vào prompt
    history_context = ""
    if chat_history_text:
        history_context = f"\nLịch sử trò chuyện gần đây:\n{chat_history_text}\n"
    
    if not context.strip():
        return f"""{history_context}Câu hỏi mới: {question}

Hướng dẫn:
1. Trả lời câu hỏi một cách tự nhiên và chính xác nhất có thể.
2. Nếu câu hỏi liên quan đến môn Lý Thuyết Thông Tin, hãy ưu tiên sử dụng kiến thức chuyên môn.
3. Nếu câu hỏi không liên quan đến môn học, hãy trả lời một cách tự nhiên và phù hợp.
4. Sử dụng định dạng markdown để làm nổi bật:
   - **In đậm** cho nội dung quan trọng
   - *In nghiêng* cho phần nhấn mạnh
   - Danh sách đánh số cho các bước/quy trình
   - Danh sách không đánh số cho liệt kê thông thường
   - Bảng cho dữ liệu có cấu trúc
5. Chỉ tham khảo lịch sử trò chuyện nếu câu hỏi hiện tại thực sự liên quan đến các câu hỏi trước đó.

Câu trả lời:"""
        
    # Tạo prompt với few-shot example, chain-of-thought và lịch sử trò chuyện
    prompt = f"""{history_context}Thông tin tham khảo từ tài liệu môn học: {context}

Câu hỏi mới: {question}

Hướng dẫn:
1. Phân tích câu hỏi và xác định xem nó có liên quan đến môn Lý Thuyết Thông Tin không.
2. Nếu liên quan đến môn học:
   - Ưu tiên sử dụng thông tin tham khảo từ tài liệu môn học
   - Kết hợp với kiến thức chuyên môn của bạn để trả lời đầy đủ và chính xác
3. Nếu không liên quan đến môn học:
   - Trả lời dựa trên kiến thức chung của bạn
   - Không cần bắt buộc sử dụng thông tin tham khảo
4. Trả lời một cách tự nhiên và phù hợp với ngữ cảnh
5. Sử dụng định dạng markdown để làm nổi bật:
   - **In đậm** cho nội dung quan trọng
   - *In nghiêng* cho phần nhấn mạnh
   - Danh sách đánh số cho các bước/quy trình
   - Danh sách không đánh số cho liệt kê thông thường
   - Bảng cho dữ liệu có cấu trúc
6. Chỉ tham khảo lịch sử trò chuyện nếu câu hỏi hiện tại thực sự liên quan đến các câu hỏi trước đó.

Câu trả lời:"""
    return prompt

def clean_expired_chats():
    """Dọn dẹp các cuộc trò chuyện đã hết hạn"""
    current_time = datetime.now()
    expired_session_ids = []
    
    for session_id, session_data in chat_history.items():
        last_updated = session_data.get('last_updated')
        if last_updated:
            time_diff = (current_time - last_updated).total_seconds() / 60
            if time_diff > CHAT_HISTORY_EXPIRATION:
                expired_session_ids.append(session_id)
    
    for session_id in expired_session_ids:
        del chat_history[session_id]
        
    logger.info(f"Đã dọn dẹp {len(expired_session_ids)} phiên trò chuyện hết hạn")

# Hàm làm sạch HTML đầu ra
def sanitize_html(html_content):
    """Làm sạch HTML để tránh các rủi ro XSS"""
    try:
        import bleach
        allowed_tags = ['p', 'ul', 'ol', 'li', 'strong', 'em', 'table', 'tr', 'td', 'th', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        allowed_attrs = {'*': ['class']}
        html_content = bleach.clean(html_content, tags=allowed_tags, attributes=allowed_attrs, strip=True)
    except ImportError:
        logger.warning("Thư viện bleach không được cài đặt, bỏ qua việc sanitize HTML")
    return html_content

def auto_title_conversation(text, conversation_id):
    """Tạo tên cho cuộc trò chuyện dựa trên nội dung câu hỏi đầu tiên"""
    try:
        # Giới hạn độ dài của prompt để tránh lỗi
        prompt = f"Tạo một tên ngắn gọn (dưới 40 ký tự) cho một cuộc trò chuyện dựa trên câu hỏi sau: '{text[:100]}...'. Chỉ trả về tiêu đề, không có giải thích hay định dạng khác."
        
        # Sử dụng Gemini để tạo tên
        response = gemini_model.generate_content(prompt)
        title = response.text.strip()
        
        # Giới hạn độ dài tiêu đề và loại bỏ dấu ngoặc kép nếu có
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        
        # Đảm bảo tiêu đề không quá dài
        if len(title) > 40:
            title = title[:37] + '...'
            
        logger.info(f"Đã tạo tiêu đề tự động: {title} cho cuộc trò chuyện: {conversation_id}")
        return title
    except Exception as e:
        logger.error(f"Lỗi khi tạo tiêu đề tự động: {str(e)}")
        return "Cuộc trò chuyện mới"

@app.route('/', methods=['POST', 'GET'])
def index():
    # Khởi tạo user_id nếu chưa có
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        user_id = session['user_id']
        user_conversations[user_id] = []
        logger.info(f"Khởi tạo user mới: {user_id}")
    else:
        user_id = session['user_id']
        if user_id not in user_conversations:
            user_conversations[user_id] = []
    
    # Khởi tạo hoặc lấy session_id hiện tại
    if 'session_id' not in session:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Thêm vào danh sách cuộc trò chuyện của user
        user_conversations[user_id].append({
            'id': session_id,
            'title': 'Cuộc trò chuyện mới',
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        })
        
        logger.info(f"Khởi tạo session mới: {session_id} cho user: {user_id}")
    else:
        session_id = session['session_id']
    
    # Khởi tạo lịch sử chat nếu chưa có
    if session_id not in chat_history:
        chat_history[session_id] = {
            'messages': [],
            'last_updated': datetime.now()
        }
    
    # Cập nhật số lượng tin nhắn cho template
    session['messages_count'] = len(chat_history[session_id]['messages'])
    
    # Đảm bảo danh sách cuộc trò chuyện được truyền cho template
    conversations = user_conversations.get(user_id, [])
    
    # Đánh dấu cuộc trò chuyện hiện tại
    current_conversation = None
    for conv in conversations:
        if conv['id'] == session_id:
            current_conversation = conv
            break
    
    # Dọn dẹp các phiên trò chuyện đã hết hạn
    clean_expired_chats()
    
    if request.method == 'POST':
        try:
            prompt = request.form['prompt']
            logger.info(f"Nhận câu hỏi: {prompt[:50]}...")
            
            # Tìm kiếm ngữ cảnh liên quan từ dữ liệu đã lưu
            context, sources_used = semantic_search_context(prompt)
            
            # Tìm kiếm câu hỏi tương tự từ kiến thức đã học
            similar_questions = learning_system.find_similar_questions(prompt)
            learned_context = ""
            if similar_questions:
                learned_context = "\nCác câu hỏi tương tự đã được trả lời:\n"
                for q in similar_questions:
                    learned_context += f"Q: {q['question']}\nA: {q['answer']}\n\n"
            
            # Lấy lịch sử trò chuyện
            chat_history_entries = chat_history[session_id]['messages']
            recent_history = chat_history_entries[-10:] if len(chat_history_entries) > 0 else []
            
            # Định dạng lịch sử cuộc trò chuyện
            chat_history_text = ""
            if recent_history:
                for idx, entry in enumerate(recent_history):
                    chat_history_text += f"Người dùng: {entry['user']}\n"
                    chat_history_text += f"Trợ lý: {entry['assistant']}\n\n"
            
            # Chuẩn bị prompt cho Gemini
            if not context.strip() and not learned_context:
                logger.info("Không tìm thấy ngữ cảnh liên quan, để Gemini tự trả lời")
                full_prompt = create_augmented_prompt(prompt, "", sources_used, chat_history_text)
            else:
                combined_context = f"{context}\n{learned_context}"
                logger.info(f"Tìm thấy ngữ cảnh ({len(combined_context)} ký tự), yêu cầu Gemini trả lời dựa trên ngữ cảnh")
                full_prompt = create_augmented_prompt(prompt, combined_context, sources_used, chat_history_text)
            
            # Gọi Gemini API
            response = gemini_model.generate_content(full_prompt)
            try:
                answer = response.candidates[0].content.parts[0].text
                
                # Xử lý trước khi chuyển đổi markdown
                # Chuẩn hóa định dạng markdown để xử lý khoảng trắng thừa
                
                # Loại bỏ khoảng trắng thừa xung quanh ký tự đánh dấu đậm và nghiêng
                # Xử lý in đậm
                answer = re.sub(r'\*\* +([^*]+) +\*\*', r'**\1**', answer)  # Loại bỏ khoảng trắng trong **text**
                answer = re.sub(r'\* +([^*]+) +\*', r'*\1*', answer)  # Loại bỏ khoảng trắng trong *text*
                
                # Xử lý in đậm với dấu gạch dưới
                answer = re.sub(r'__ +([^_]+) +__', r'__\1__', answer)  # Loại bỏ khoảng trắng trong __text__
                answer = re.sub(r'_ +([^_]+) +_', r'_\1_', answer)  # Loại bỏ khoảng trắng trong _text_
                
                # Đảm bảo danh sách được định dạng đúng
                lines = answer.split('\n')
                for i in range(len(lines)):
                    # Nếu dòng bắt đầu bằng dấu gạch đầu dòng hoặc số, đảm bảo có khoảng trắng sau dấu
                    if lines[i].strip().startswith('*') or lines[i].strip().startswith('-'):
                        if not lines[i].strip().startswith('* ') and not lines[i].strip().startswith('- '):
                            lines[i] = lines[i].replace('*', '* ', 1).replace('-', '- ', 1)
                    # Xử lý danh sách có thứ tự
                    elif len(lines[i].strip()) > 2 and lines[i].strip()[0].isdigit() and lines[i].strip()[1] == '.':
                        if not lines[i].strip()[2:].startswith(' '):
                            parts = lines[i].strip().split('.', 1)
                            if len(parts) > 1:
                                lines[i] = parts[0] + '. ' + parts[1]
                
                # Xử lý trước các bảng Markdown
                in_table = False
                table_text = []
                processed_lines = []
                
                for line in lines:
                    if '|' in line and not in_table and line.strip().startswith('|'):
                        in_table = True
                        table_text.append(line)
                    elif in_table and ('|' in line):
                        table_text.append(line)
                    elif in_table:
                        # Điều chỉnh định dạng bảng nếu cần thiết
                        if len(table_text) > 2:  # Đảm bảo ít nhất có hàng tiêu đề, định dạng và một hàng dữ liệu
                            processed_table = '\n'.join(table_text)
                            processed_lines.append(processed_table)
                        else:
                            processed_lines.extend(table_text)
                        processed_lines.append(line)
                        in_table = False
                        table_text = []
                    else:
                        processed_lines.append(line)
                
                # Thêm bảng cuối cùng nếu có
                if in_table and len(table_text) > 0:
                    if len(table_text) > 2:  # Đảm bảo ít nhất có hàng tiêu đề, định dạng và một hàng dữ liệu
                        processed_table = '\n'.join(table_text)
                        processed_lines.append(processed_table)
                    else:
                        processed_lines.extend(table_text)
                
                answer = '\n'.join(processed_lines)
                
                # Chuyển đổi đánh dấu markdown thành HTML
                answer = markdown.markdown(answer, extensions=['tables'])
                
                # Làm sạch HTML đầu ra (phòng chống XSS)
                answer = sanitize_html(answer)
                
                # Làm sạch HTML đầu ra
                # Loại bỏ khoảng trắng thừa trong thẻ <strong> và <em>
                answer = re.sub(r'<strong>\s+', r'<strong>', answer)
                answer = re.sub(r'\s+</strong>', r'</strong>', answer)
                answer = re.sub(r'<em>\s+', r'<em>', answer)
                answer = re.sub(r'\s+</em>', r'</em>', answer)
                
                # Thêm các lớp và thuộc tính để cải thiện hiển thị phân cấp
                answer = answer.replace('<ul>', '<ul class="nested-list">')
                answer = answer.replace('<ol>', '<ol class="nested-list">')
                
                # Thêm một số điều chỉnh để đảm bảo danh sách hiển thị đúng
                answer = answer.replace('<ul>\n', '<ul>')
                answer = answer.replace('\n</ul>', '</ul>')
                answer = answer.replace('<ol>\n', '<ol>')
                answer = answer.replace('\n</ol>', '</ol>')
                answer = answer.replace('<li>\n', '<li>')
                answer = answer.replace('\n</li>', '</li>')
                
                # Điều chỉnh bảng
                answer = answer.replace('<table>', '<table class="table custom-table">')
                
                # Loại bỏ triệt để các thẻ <br>, <p> hoặc thẻ rỗng trước bảng
                answer = re.sub(r'((<br>|<p>\s*</p>|\s|<p><br></p>|<p></p>|<p>\s*</p>)+)<table', '<table', answer)
                answer = re.sub(r'(<div>\s*</div>)+<table', '<table', answer)
                
                # Đảm bảo căn chỉnh trong các ô bảng
                answer = re.sub(r'<td>\s+', '<td>', answer)
                answer = re.sub(r'\s+</td>', '</td>', answer)
                answer = re.sub(r'<th>\s+', '<th>', answer)
                answer = re.sub(r'\s+</th>', '</th>', answer)
                
                # Thay thế các dấu xuống dòng bằng thẻ <br>
                answer = answer.replace('\n', '<br>')
                
                # Log kết quả
                logger.info(f"Gemini trả lời thành công ({len(answer)} ký tự HTML)")
                
                # Cập nhật lịch sử cuộc trò chuyện
                plain_answer = re.sub(r'<[^>]*>', '', answer)  # Loại bỏ HTML tags để lưu text thuần túy
                chat_history[session_id]['messages'].append({
                    'user': prompt,
                    'assistant': plain_answer[:500]  # Lưu tối đa 500 ký tự để tiết kiệm bộ nhớ
                })
                chat_history[session_id]['last_updated'] = datetime.now()
                
                # Tự động đặt tên cho cuộc trò chuyện nếu là tin nhắn đầu tiên
                if len(chat_history[session_id]['messages']) == 1 and 'user_id' in session:
                    user_id = session['user_id']
                    for conv in user_conversations.get(user_id, []):
                        if conv['id'] == session_id and conv['title'] == 'Cuộc trò chuyện mới':
                            # Đặt tên tự động cho cuộc trò chuyện
                            new_title = auto_title_conversation(prompt, session_id)
                            conv['title'] = new_title
                            break
                
                # Giới hạn lịch sử trò chuyện để tiết kiệm bộ nhớ (tối đa 20 tương tác)
                if len(chat_history[session_id]['messages']) > 20:
                    chat_history[session_id]['messages'] = chat_history[session_id]['messages'][-20:]
                
                # Học từ cuộc trò chuyện này
                learning_system.learn_from_conversation(prompt, plain_answer[:500])
                
            except Exception as e:
                error_msg = f"Không thể lấy nội dung trả lời từ Gemini: {str(e)}"
                logger.error(error_msg)
                answer = f"Không thể lấy nội dung trả lời từ Gemini!"
                
            return answer
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    return render_template('index.html', **locals())

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Endpoint cho phép xóa lịch sử trò chuyện hiện tại"""
    if 'session_id' in session:
        session_id = session['session_id']
        if session_id in chat_history:
            chat_history[session_id]['messages'] = []
            logger.info(f"Đã xóa lịch sử trò chuyện cho session: {session_id}")
            # Cập nhật số lượng tin nhắn
            session['messages_count'] = 0
            
            # Cập nhật tiêu đề cuộc trò chuyện
            if 'user_id' in session:
                user_id = session['user_id']
                for conv in user_conversations.get(user_id, []):
                    if conv['id'] == session_id:
                        conv['title'] = 'Cuộc trò chuyện mới'
                        conv['last_updated'] = datetime.now().isoformat()
                        break
    
    return redirect(url_for('index'))

@app.route('/new_conversation', methods=['POST', 'GET'])
def new_conversation():
    """Tạo cuộc trò chuyện mới"""
    # Tạo UUID mới cho session
    new_session_id = str(uuid.uuid4())
    
    # Lưu vào session
    old_session_id = session.get('session_id')
    session['session_id'] = new_session_id
    
    # Khởi tạo lịch sử trò chuyện mới
    chat_history[new_session_id] = {
        'messages': [],
        'last_updated': datetime.now()
    }
    
    # Cập nhật số lượng tin nhắn
    session['messages_count'] = 0
    
    # Thêm vào danh sách cuộc trò chuyện của user
    if 'user_id' in session:
        user_id = session['user_id']
        if user_id in user_conversations:
            user_conversations[user_id].append({
                'id': new_session_id,
                'title': 'Cuộc trò chuyện mới',
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            })
    
    logger.info(f"Đã tạo cuộc trò chuyện mới với ID: {new_session_id}")
    
    return redirect(url_for('index'))

@app.route('/switch_conversation/<conversation_id>', methods=['GET'])
def switch_conversation(conversation_id):
    """Chuyển đổi giữa các cuộc trò chuyện"""
    if 'user_id' in session:
        user_id = session['user_id']
        # Kiểm tra xem conversation_id có thuộc về user hiện tại không
        valid_conversation = False
        for conv in user_conversations.get(user_id, []):
            if conv['id'] == conversation_id:
                valid_conversation = True
                # Cập nhật thời gian truy cập
                conv['last_updated'] = datetime.now().isoformat()
                break
        
        if valid_conversation:
            # Lưu session_id mới
            session['session_id'] = conversation_id
            
            # Khởi tạo lịch sử chat nếu chưa có
            if conversation_id not in chat_history:
                chat_history[conversation_id] = {
                    'messages': [],
                    'last_updated': datetime.now()
                }
            
            # Cập nhật số lượng tin nhắn
            session['messages_count'] = len(chat_history.get(conversation_id, {}).get('messages', []))
            
            logger.info(f"Đã chuyển sang cuộc trò chuyện: {conversation_id}")
    
    return redirect(url_for('index'))

@app.route('/rename_conversation/<conversation_id>', methods=['POST', 'GET'])
def rename_conversation(conversation_id):
    """Đổi tên cuộc trò chuyện"""
    if request.method == 'POST':
        new_title = request.form.get('title', 'Cuộc trò chuyện không tên')
        
        if 'user_id' in session:
            user_id = session['user_id']
            # Kiểm tra xem conversation_id có thuộc về user hiện tại không
            for conv in user_conversations.get(user_id, []):
                if conv['id'] == conversation_id:
                    conv['title'] = new_title
                    logger.info(f"Đã đổi tên cuộc trò chuyện {conversation_id} thành: {new_title}")
                    break
    
    return redirect(url_for('index'))

@app.route('/delete_conversation/<conversation_id>', methods=['POST', 'GET'])
def delete_conversation(conversation_id):
    """Xóa một cuộc trò chuyện"""
    if 'user_id' in session:
        user_id = session['user_id']
        
        # Lọc ra các cuộc trò chuyện không bị xóa
        filtered_conversations = []
        for conv in user_conversations.get(user_id, []):
            if conv['id'] != conversation_id:
                filtered_conversations.append(conv)
        
        user_conversations[user_id] = filtered_conversations
        
        # Xóa lịch sử trò chuyện
        if conversation_id in chat_history:
            del chat_history[conversation_id]
        
        # Nếu cuộc trò chuyện hiện tại bị xóa, chuyển sang cuộc trò chuyện mới
        if session.get('session_id') == conversation_id:
            if filtered_conversations:
                # Chuyển sang cuộc trò chuyện gần nhất
                session['session_id'] = filtered_conversations[-1]['id']
                session['messages_count'] = len(chat_history.get(filtered_conversations[-1]['id'], {}).get('messages', []))
            else:
                # Tạo cuộc trò chuyện mới nếu không còn cuộc trò chuyện nào
                return redirect(url_for('new_conversation'))
        
        logger.info(f"Đã xóa cuộc trò chuyện: {conversation_id}")
    
    return redirect(url_for('index'))

@app.route('/get_conversations', methods=['GET'])
def get_conversations():
    """Endpoint để lấy danh sách các cuộc trò chuyện của người dùng hiện tại"""
    if 'user_id' in session:
        user_id = session['user_id']
        conversations = user_conversations.get(user_id, [])
        current_id = session.get('session_id')
        
        # Thêm trường active để đánh dấu cuộc trò chuyện hiện tại
        for conv in conversations:
            conv['active'] = (conv['id'] == current_id)
            
            # Nếu là cuộc trò chuyện hiện tại, thêm số tin nhắn
            if conv['id'] == current_id:
                message_count = len(chat_history.get(current_id, {}).get('messages', []))
                conv['messages_count'] = message_count
        
        return jsonify({'conversations': conversations})
    
    return jsonify({'conversations': []})

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    """Endpoint để lấy lịch sử trò chuyện cho người dùng hiện tại"""
    if 'session_id' in session:
        session_id = session['session_id']
        if session_id in chat_history:
            # Trả về danh sách tin nhắn dạng JSON
            messages = chat_history[session_id]['messages']
            return jsonify({'history': messages})
    
    # Nếu không có lịch sử hoặc không có session, trả về danh sách trống
    return jsonify({'history': []})

if __name__ == '__main__':
    # Trong môi trường phát triển, debug=True, trong production nên đặt debug=False
    is_production = os.environ.get('FLASK_ENV') == 'production'
    app.run(debug=not is_production, host='127.0.0.1', port=int(os.environ.get('PORT', 5000)))
