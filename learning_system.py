import json
import os
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

class LearningSystem:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.knowledge_file = 'learned_knowledge.json'
        self.embeddings_file = 'learned_embeddings.npy'
        self.load_knowledge()

    def load_knowledge(self):
        """Load kiến thức đã học từ file"""
        if os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        else:
            self.knowledge_base = []

        if os.path.exists(self.embeddings_file):
            self.embeddings = np.load(self.embeddings_file)
        else:
            self.embeddings = np.array([])

    def save_knowledge(self):
        """Lưu kiến thức đã học vào file"""
        with open(self.knowledge_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        np.save(self.embeddings_file, self.embeddings)

    def learn_from_conversation(self, question, answer):
        """Học từ một cuộc trò chuyện"""
        # Tạo embedding cho câu hỏi
        question_embedding = self.embedding_model.encode(question, convert_to_numpy=True)
        
        # Thêm vào knowledge base
        new_knowledge = {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat(),
            'embedding_index': len(self.knowledge_base)
        }
        self.knowledge_base.append(new_knowledge)
        
        # Cập nhật embeddings
        if len(self.embeddings) == 0:
            self.embeddings = question_embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, question_embedding])
        
        # Lưu kiến thức mới
        self.save_knowledge()

    def find_similar_questions(self, question, top_k=3):
        """Tìm các câu hỏi tương tự từ kiến thức đã học"""
        if len(self.knowledge_base) == 0:
            return []

        # Tạo embedding cho câu hỏi mới
        question_embedding = self.embedding_model.encode(question, convert_to_numpy=True)
        
        # Tính độ tương đồng cosine
        scores = np.dot(self.embeddings, question_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-8)
        
        # Lấy top_k câu hỏi tương tự nhất
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        similar_questions = []
        for idx in top_indices:
            if scores[idx] > 0.3:  # Chỉ lấy những câu có độ tương đồng > 0.3
                similar_questions.append({
                    'question': self.knowledge_base[idx]['question'],
                    'answer': self.knowledge_base[idx]['answer'],
                    'similarity_score': float(scores[idx])
                })
        
        return similar_questions 