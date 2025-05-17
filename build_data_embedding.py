import os
import pickle
from sentence_transformers import SentenceTransformer
import re

# Cấu hình model embedding
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def chunk_text_files(filepaths, chunk_size=3, overlap=1):
    all_chunks = []
    chunk_sources = []
    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            # Tách thành các câu
            sentences = re.split(r'(?<=[.!?])\s+', content)
            sentences = [s for s in sentences if s.strip()]
            for i in range(0, len(sentences), chunk_size - overlap):
                if i + chunk_size <= len(sentences):
                    chunk = " ".join(sentences[i:i + chunk_size])
                    all_chunks.append(chunk)
                    chunk_sources.append(filepath)
                elif i < len(sentences):
                    chunk = " ".join(sentences[i:])
                    all_chunks.append(chunk)
                    chunk_sources.append(filepath)
        except Exception as e:
            print(f"Lỗi khi đọc file {filepath}: {str(e)}")
            continue
    return all_chunks, chunk_sources

def build_and_save_embedding(data_dir='DATA', output_file='embedding_data.pkl'):
    # Lấy tất cả file .txt và .tex
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt') or f.endswith('.tex')]
    print(f"Đang xử lý {len(filepaths)} file trong {data_dir}...")
    chunks, sources = chunk_text_files(filepaths)
    print(f"Tổng số chunk: {len(chunks)}")
    # Tính embedding
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    # Lưu ra file
    with open(output_file, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'chunks': chunks, 'sources': sources}, f)
    print(f"Đã lưu embedding vào {output_file}")

if __name__ == '__main__':
    build_and_save_embedding() 