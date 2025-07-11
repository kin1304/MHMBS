import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from .beam_search import BeamSearchPathFinder
import unicodedata
import re
from difflib import SequenceMatcher
from typing import List, Dict

try:
    from mint.helpers import segment_entity_with_vncorenlp
except ImportError:
    try:
        from process_with_beam_search_fixed import segment_entity_with_vncorenlp
    except ImportError:
        segment_entity_with_vncorenlp = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

class TextGraph:
    """
    Lớp TextGraph để xây dựng và phân tích đồ thị văn bản từ context và claim
    
    Đồ thị bao gồm các loại node:
    - Word nodes: chứa từng từ trong context và claim
    - Sentence nodes: các câu trong context  
    - Claim node: giá trị claim
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.word_nodes = {}
        self.sentence_nodes = {}
        self.claim_node = None
        self.entity_nodes = {}  # Thêm dictionary để quản lý entity nodes
        
        # POS tag filtering configuration
        self.enable_pos_filtering = True  # Mặc định bật để giảm nhiễu
        self.important_pos_tags = {
            'N',    # Danh từ thường
            'Np',   # Danh từ riêng
            'V',    # Động từ
            'A',    # Tính từ
            'Nc',   # Danh từ chỉ người
            'M',    # Số từ
            'R',    # Trạng từ (có thể tranh luận)
            'P'     # Đại từ (có thể tranh luận)
        }
        
        # Load environment variables
        load_dotenv()
        self.openai_client = None
        self._init_openai_client()
        
        # Semantic similarity components
        self.phobert_tokenizer = None
        self.phobert_model = None
        self.word_embeddings = {}  # Cache embeddings
        self.embedding_dim = 768  # PhoBERT base dimension (full dimension - no PCA)
        self.faiss_index = None
        self.word_to_index = {}  # Mapping từ word -> index trong faiss
        self.index_to_word = {}  # Mapping ngược lại
        
        # Semantic similarity parameters (optimized for full embeddings)
        self.similarity_threshold = 0.85
        self.top_k_similar = 5
        
        self._init_phobert_model()
    
    def set_pos_filtering(self, enable=True, custom_pos_tags=None):
        """
        Cấu hình lọc từ loại cho word nodes
        
        Args:
            enable (bool): Bật/tắt tính năng lọc từ loại
            custom_pos_tags (set): Tập hợp các từ loại muốn giữ lại (nếu None thì dùng mặc định)
        """
        self.enable_pos_filtering = enable
        if custom_pos_tags is not None:
            self.important_pos_tags = set(custom_pos_tags)
    
    def is_important_word(self, word, pos_tag):
        """
        Kiểm tra xem từ có quan trọng hay không dựa trên từ loại
        
        Args:
            word (str): Từ cần kiểm tra
            pos_tag (str): Từ loại của từ
            
        Returns:
            bool: True nếu từ quan trọng và nên tạo word node
        """
        # Nếu không bật lọc từ loại, tất cả từ đều quan trọng
        if not self.enable_pos_filtering:
            return True
            
        # Kiểm tra từ loại có trong danh sách quan trọng không
        return pos_tag in self.important_pos_tags
    
    def add_word_node(self, word, pos_tag=None, lemma=None):
        """Thêm word node vào đồ thị (có thể lọc theo từ loại)"""
        # Kiểm tra xem từ có quan trọng không
        if not self.is_important_word(word, pos_tag):
            return None  # Không tạo node cho từ không quan trọng
            
        if word not in self.word_nodes:
            node_id = f"word_{len(self.word_nodes)}"
            self.word_nodes[word] = node_id
            self.graph.add_node(node_id, 
                              type="word", 
                              text=word, 
                              pos=pos_tag, 
                              lemma=lemma)
        return self.word_nodes[word]
    
    def add_sentence_node(self, sentence_id, sentence_text):
        """Thêm sentence node vào đồ thị"""
        node_id = f"sentence_{sentence_id}"
        self.sentence_nodes[sentence_id] = node_id
        self.graph.add_node(node_id, 
                          type="sentence", 
                          text=sentence_text)
        return node_id
    
    def add_claim_node(self, claim_text):
        """Thêm claim node vào đồ thị"""
        self.claim_node = "claim_0"
        self.graph.add_node(self.claim_node, 
                          type="claim", 
                          text=claim_text)
        return self.claim_node
    
    def connect_word_to_sentence(self, word_node, sentence_node):
        """Kết nối word với sentence"""
        self.graph.add_edge(word_node, sentence_node, relation="belongs_to", edge_type="structural")
    
    def connect_word_to_claim(self, word_node, claim_node):
        """Kết nối word với claim"""
        self.graph.add_edge(word_node, claim_node, relation="belongs_to", edge_type="structural")
    
    def connect_dependency(self, dependent_word_node, head_word_node, dep_label):
        """Kết nối dependency giữa hai từ"""
        self.graph.add_edge(dependent_word_node, head_word_node, 
                          relation=dep_label, edge_type="dependency")
    
    def build_from_vncorenlp_output(self, context_sentences, claim_text, claim_sentences):
        """Xây dựng đồ thị từ kết quả py_vncorenlp"""
        
        # Thêm claim node
        claim_node = self.add_claim_node(claim_text)
        
        # Xử lý các câu trong context (context_sentences là dict)
        for sent_idx, sentence_tokens in context_sentences.items():
            sentence_text = " ".join([token["wordForm"] for token in sentence_tokens])
            sentence_node = self.add_sentence_node(sent_idx, sentence_text)
            
            # Dictionary để map index -> word_node_id cho việc tạo dependency links
            token_index_to_node = {}
            
            # Thêm các word trong sentence
            for token in sentence_tokens:
                word = token["wordForm"]
                pos_tag = token.get("posTag", "")
                lemma = token.get("lemma", "")
                token_index = token.get("index", 0)
                
                word_node = self.add_word_node(word, pos_tag, lemma)
                
                # Chỉ tạo kết nối nếu word_node được tạo thành công (không bị lọc)
                if word_node is not None:
                    self.connect_word_to_sentence(word_node, sentence_node)
                    # Lưu mapping để tạo dependency links sau
                    token_index_to_node[token_index] = word_node
            
            # Tạo dependency connections giữa các từ trong câu
            for token in sentence_tokens:
                token_index = token.get("index", 0)
                head_index = token.get("head", 0)
                dep_label = token.get("depLabel", "")
                
                # Chỉ tạo dependency nếu cả dependent và head đều tồn tại trong mapping
                if (head_index > 0 and 
                    token_index in token_index_to_node and 
                    head_index in token_index_to_node):
                    dependent_node = token_index_to_node[token_index]
                    head_node = token_index_to_node[head_index]
                    self.connect_dependency(dependent_node, head_node, dep_label)
        
        # Xử lý các word trong claim (claim_sentences cũng là dict)
        for sent_idx, sentence_tokens in claim_sentences.items():
            # Dictionary để map index -> word_node_id cho claim
            claim_token_index_to_node = {}
            
            # Thêm words
            for token in sentence_tokens:
                word = token["wordForm"]
                pos_tag = token.get("posTag", "")
                lemma = token.get("lemma", "")
                token_index = token.get("index", 0)
                
                word_node = self.add_word_node(word, pos_tag, lemma)
                
                # Chỉ tạo kết nối nếu word_node được tạo thành công (không bị lọc)
                if word_node is not None:
                    self.connect_word_to_claim(word_node, claim_node)
                    # Lưu mapping cho dependency links
                    claim_token_index_to_node[token_index] = word_node
            
            # Tạo dependency connections trong claim
            for token in sentence_tokens:
                token_index = token.get("index", 0)
                head_index = token.get("head", 0)
                dep_label = token.get("depLabel", "")
                
                # Chỉ tạo dependency nếu cả dependent và head đều tồn tại trong mapping
                if (head_index > 0 and 
                    token_index in claim_token_index_to_node and 
                    head_index in claim_token_index_to_node):
                    dependent_node = claim_token_index_to_node[token_index]
                    head_node = claim_token_index_to_node[head_index]
                    self.connect_dependency(dependent_node, head_node, dep_label)
    
    def get_statistics(self):
        """Thống kê cơ bản về đồ thị"""
        word_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'word'])
        sentence_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'sentence'])
        claim_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'claim'])
        entity_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'entity'])
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "word_nodes": word_count,
            "sentence_nodes": sentence_count,
            "claim_nodes": claim_count,
            "entity_nodes": entity_count
        }
    
    def get_shared_words(self):
        """Tìm các từ xuất hiện cả trong context và claim"""
        shared_words = []
        
        for word_node_id in self.word_nodes.values():
            # Kiểm tra xem word node có kết nối với cả sentence nodes và claim node không
            neighbors = list(self.graph.neighbors(word_node_id))
            has_sentence_connection = any(
                self.graph.nodes[neighbor]['type'] == 'sentence' for neighbor in neighbors
            )
            has_claim_connection = any(
                self.graph.nodes[neighbor]['type'] == 'claim' for neighbor in neighbors
            )
            
            if has_sentence_connection and has_claim_connection:
                word_text = self.graph.nodes[word_node_id]['text']
                pos_tag = self.graph.nodes[word_node_id]['pos']
                shared_words.append({
                    'word': word_text,
                    'pos': pos_tag,
                    'node_id': word_node_id
                })
        
        return shared_words
    
    def get_word_frequency(self):
        """Đếm tần suất xuất hiện của từng từ"""
        word_freq = {}
        for word_node_id in self.word_nodes.values():
            word_text = self.graph.nodes[word_node_id]['text']
            word_freq[word_text] = word_freq.get(word_text, 0) + 1
        return word_freq
    
    def get_dependency_statistics(self):
        """Thống kê về các mối quan hệ dependency"""
        dependency_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'dependency'
        ]
        
        # Đếm các loại dependency
        dep_types = {}
        for u, v, data in dependency_edges:
            dep_label = data.get('relation', 'unknown')
            dep_types[dep_label] = dep_types.get(dep_label, 0) + 1
        
        return {
            "total_dependency_edges": len(dependency_edges),
            "dependency_types": dep_types,
            "most_common_dependencies": sorted(dep_types.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def get_word_dependencies(self, word):
        """Lấy tất cả dependencies của một từ"""
        if word not in self.word_nodes:
            return {"dependents": [], "heads": []}
        
        word_node_id = self.word_nodes[word]
        dependents = []
        heads = []
        
        for neighbor in self.graph.neighbors(word_node_id):
            edge_data = self.graph.edges[word_node_id, neighbor]
            if edge_data.get('edge_type') == 'dependency':
                dep_relation = edge_data.get('relation', '')
                neighbor_word = self.graph.nodes[neighbor]['text']
                
                # Kiểm tra xem word_node_id là head hay dependent
                # Trong NetworkX undirected graph, cần kiểm tra hướng dựa trên semantic
                # Giả sử edge được tạo từ dependent -> head
                if (word_node_id, neighbor) in self.graph.edges():
                    heads.append({"word": neighbor_word, "relation": dep_relation})
                else:
                    dependents.append({"word": neighbor_word, "relation": dep_relation})
        
        return {"dependents": dependents, "heads": heads}
    
    def get_detailed_statistics(self):
        """Thống kê chi tiết về đồ thị"""
        basic_stats = self.get_statistics()
        shared_words = self.get_shared_words()
        word_freq = self.get_word_frequency()
        dep_stats = self.get_dependency_statistics()
        semantic_stats = self.get_semantic_statistics()
        
        # Tìm từ xuất hiện nhiều nhất
        most_frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Tính tổng edges theo loại
        structural_edges = len([
            (u, v) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'structural'
        ])
        
        entity_structural_edges = len([
            (u, v) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'entity_structural'
        ])
        
        sentence_claim_semantic_edges = len([
            (u, v) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'sentence_claim_semantic'
        ])
        
        # Thống kê entity
        entity_list = [
            {
                'name': self.graph.nodes[node_id]['text'],
                'type': self.graph.nodes[node_id].get('entity_type', 'ENTITY'),
                'connected_sentences': len([
                    neighbor for neighbor in self.graph.neighbors(node_id) 
                    if self.graph.nodes[neighbor]['type'] == 'sentence'
                ])
            }
            for node_id in self.graph.nodes() 
            if self.graph.nodes[node_id]['type'] == 'entity'
        ]
        
        # Thống kê sentence-claim semantic edges
        sentence_claim_stats = self.get_sentence_claim_semantic_statistics()
        
        return {
            **basic_stats,
            "shared_words_count": len(shared_words),
            "shared_words": shared_words,
            "unique_words": len(word_freq),
            "most_frequent_words": most_frequent_words,
            "average_words_per_sentence": basic_stats['word_nodes'] / max(basic_stats['sentence_nodes'], 1),
            "dependency_statistics": dep_stats,
            "structural_edges": structural_edges,
            "dependency_edges": dep_stats["total_dependency_edges"],
            "entity_structural_edges": entity_structural_edges,
            "sentence_claim_semantic_edges": sentence_claim_semantic_edges,
            "entities": entity_list,
            "unique_entities": len(entity_list),
            "semantic_statistics": semantic_stats,
            "semantic_edges": semantic_stats["total_semantic_edges"],
            "sentence_claim_semantic_statistics": sentence_claim_stats
        }
    
    def visualize(self, figsize=(15, 10), show_dependencies=True, show_semantic=True):
        """Vẽ đồ thị với phân biệt structural, dependency, entity và semantic edges"""
        plt.figure(figsize=figsize)
        
        # Định nghĩa màu sắc cho các loại node
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node]['type']
            if node_type == 'word':
                node_colors.append('lightblue')
                node_sizes.append(200)
            elif node_type == 'sentence':
                node_colors.append('lightgreen')
                node_sizes.append(500)
            elif node_type == 'claim':
                node_colors.append('lightcoral')
                node_sizes.append(600)
            elif node_type == 'entity':
                node_colors.append('gold')
                node_sizes.append(400)
        
        # Tạo layout
        pos = nx.spring_layout(self.graph, k=2, iterations=100)
        
        # Phân chia edges theo loại
        structural_edges = []
        dependency_edges = []
        entity_edges = []
        semantic_edges = []
        sentence_claim_semantic_edges = []
        
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'structural')
            if edge_type == 'structural':
                structural_edges.append((u, v))
            elif edge_type == 'dependency':
                dependency_edges.append((u, v))
            elif edge_type == 'entity_structural':
                entity_edges.append((u, v))
            elif edge_type == 'semantic':
                semantic_edges.append((u, v))
            elif edge_type == 'sentence_claim_semantic':
                sentence_claim_semantic_edges.append((u, v))
        
        # Vẽ nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.8)
        
        # Vẽ structural edges (word -> sentence/claim)
        if structural_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=structural_edges,
                                 edge_color='gray',
                                 style='-',
                                 width=1,
                                 alpha=0.6)
        
        # Vẽ entity edges (entity -> sentence)
        if entity_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=entity_edges,
                                 edge_color='orange',
                                 style='-',
                                 width=2,
                                 alpha=0.7)
        
        # Vẽ semantic edges (word -> word)
        if show_semantic and semantic_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=semantic_edges,
                                 edge_color='purple',
                                 style=':',
                                 width=1.5,
                                 alpha=0.8)
        
        # Vẽ sentence-claim semantic edges (sentence -> claim)
        if sentence_claim_semantic_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=sentence_claim_semantic_edges,
                                 edge_color='blue',
                                 style='-',
                                 width=3,
                                 alpha=0.9)
        
        # Vẽ dependency edges (word -> word)
        if show_dependencies and dependency_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=dependency_edges,
                                 edge_color='red',
                                 style='--',
                                 width=0.8,
                                 alpha=0.7,
                                 arrows=True,
                                 arrowsize=10)
        
        # Thêm legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Word nodes'),
            mpatches.Patch(color='lightgreen', label='Sentence nodes'),
            mpatches.Patch(color='lightcoral', label='Claim node'),
            mpatches.Patch(color='gold', label='Entity nodes')
        ]
        
        edge_legend = []
        if structural_edges:
            edge_legend.append(Line2D([0], [0], color='gray', label='Structural edges'))
        if entity_edges:
            edge_legend.append(Line2D([0], [0], color='orange', label='Entity edges'))
        if sentence_claim_semantic_edges:
            edge_legend.append(Line2D([0], [0], color='blue', linewidth=3, label='Sentence-Claim Semantic'))
        if show_semantic and semantic_edges:
            edge_legend.append(Line2D([0], [0], color='purple', linestyle=':', label='Word Semantic edges'))
        if show_dependencies and dependency_edges:
            edge_legend.append(Line2D([0], [0], color='red', linestyle='--', label='Dependency edges'))
        
        legend_elements.extend(edge_legend)
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        title = f"Text Graph: Words, Sentences, Claim, Entities ({len(self.entity_nodes)} entities)"
        if sentence_claim_semantic_edges:
            title += f", S-C Semantic ({len(sentence_claim_semantic_edges)} edges)"
        if show_semantic and semantic_edges:
            title += f", Word Semantic ({len(semantic_edges)} edges)"
        if show_dependencies and dependency_edges:
            title += f", Dependencies ({len(dependency_edges)} edges)"
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_dependencies_only(self, figsize=(12, 8)):
        """Vẽ chỉ dependency graph giữa các từ"""
        # Tạo subgraph chỉ với word nodes và dependency edges
        word_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'word']
        dependency_edges = [
            (u, v) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'dependency'
        ]
        
        if not dependency_edges:
            print("Không có dependency edges để vẽ!")
            return
        
        # Tạo subgraph
        subgraph = self.graph.edge_subgraph(dependency_edges).copy()
        
        plt.figure(figsize=figsize)
        
        # Layout cho dependency graph
        pos = nx.spring_layout(subgraph, k=1.5, iterations=100)
        
        # Vẽ nodes với labels
        nx.draw_networkx_nodes(subgraph, pos, 
                             node_color='lightblue',
                             node_size=300,
                             alpha=0.8)
        
        # Vẽ edges với labels
        nx.draw_networkx_edges(subgraph, pos,
                             edge_color='red',
                             style='-',
                             width=1.5,
                             alpha=0.7,
                             arrows=True,
                             arrowsize=15)
        
        # Thêm node labels (từ)
        node_labels = {node: self.graph.nodes[node]['text'][:10] 
                      for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, node_labels, font_size=8)
        
        # Thêm edge labels (dependency relations)
        edge_labels = {(u, v): data.get('relation', '') 
                      for u, v, data in subgraph.edges(data=True)}
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=6)
        
        plt.title(f"Dependency Graph ({len(dependency_edges)} dependencies)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def save_graph(self, filepath):
        """Lưu đồ thị vào file"""
        # Đảm bảo lưu file vào thư mục gốc của project
        if not os.path.isabs(filepath):
            # Lấy thư mục cha của thư mục mint
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(project_root, filepath)
        
        # Tạo một bản copy của graph để xử lý None values
        graph_copy = self.graph.copy()
        
        # Xử lý None values trong node attributes
        for node_id in graph_copy.nodes():
            node_data = graph_copy.nodes[node_id]
            for key, value in node_data.items():
                if value is None:
                    graph_copy.nodes[node_id][key] = ""
        
        # Xử lý None values trong edge attributes
        for u, v in graph_copy.edges():
            edge_data = graph_copy.edges[u, v]
            for key, value in edge_data.items():
                if value is None:
                    graph_copy.edges[u, v][key] = ""
        
        nx.write_gexf(graph_copy, filepath)
        print(f"Đồ thị đã được lưu vào: {filepath}")
    
    def load_graph(self, filepath):
        """Tải đồ thị từ file"""
        self.graph = nx.read_gexf(filepath)
        
        # Rebuild node mappings
        self.word_nodes = {}
        self.sentence_nodes = {}
        self.entity_nodes = {}
        self.claim_node = None
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if node_data['type'] == 'word':
                self.word_nodes[node_data['text']] = node_id
            elif node_data['type'] == 'sentence':
                # Extract sentence index from node_id
                sent_idx = int(node_id.split('_')[1])
                self.sentence_nodes[sent_idx] = node_id
            elif node_data['type'] == 'claim':
                self.claim_node = node_id
            elif node_data['type'] == 'entity':
                self.entity_nodes[node_data['text']] = node_id
        
        print(f"Đồ thị đã được tải từ: {filepath}")
    
    def export_to_json(self):
        """Xuất đồ thị ra định dạng JSON để dễ dàng phân tích"""
        graph_data = {
            "nodes": [],
            "edges": [],
            "statistics": self.get_detailed_statistics()
        }
        
        # Export nodes
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            graph_data["nodes"].append({
                "id": node_id,
                "type": node_data["type"],
                "text": node_data["text"],
                "pos": node_data.get("pos", ""),
                "lemma": node_data.get("lemma", "")
            })
        
        # Export edges
        for edge in self.graph.edges():
            edge_data = self.graph.edges[edge]
            graph_data["edges"].append({
                "source": edge[0],
                "target": edge[1],
                "relation": edge_data.get("relation", ""),
                "edge_type": edge_data.get("edge_type", "")
            })
        
        return json.dumps(graph_data, ensure_ascii=False, indent=2)
    
    def _init_openai_client(self):
        """Khởi tạo OpenAI client"""
        try:
            # Try multiple key names for backward compatibility
            api_key = os.getenv('OPENAI_KEY') or os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'your_openai_api_key_here':
                self.openai_client = OpenAI(api_key=api_key)
            else:
                print("Warning: OPENAI_KEY hoặc OPENAI_API_KEY không được tìm thấy trong .env file.")
                print("Vui lòng tạo file .env và thêm OPENAI_KEY=your_api_key")
        except Exception as e:
            print(f"Lỗi khi khởi tạo OpenAI client: {e}")
    
    def add_entity_node(self, entity_name, entity_type="ENTITY"):
        """Thêm entity node vào đồ thị"""
        if entity_name not in self.entity_nodes:
            node_id = f"entity_{len(self.entity_nodes)}"
            self.entity_nodes[entity_name] = node_id
            self.graph.add_node(node_id, 
                              type="entity", 
                              text=entity_name,
                              entity_type=entity_type)
        return self.entity_nodes[entity_name]
    
    def connect_entity_to_sentence(self, entity_node, sentence_node):
        """Kết nối entity với sentence"""
        self.graph.add_edge(entity_node, sentence_node, relation="mentioned_in", edge_type="entity_structural")
    
    def _update_openai_model(self, model=None, temperature=None, max_tokens=None):
        """Update OpenAI model parameters"""
        if model:
            self.openai_model = model
        if temperature is not None:
            self.openai_temperature = temperature  
        if max_tokens is not None:
            self.openai_max_tokens = max_tokens
    
    def extract_entities_with_openai(self, context_text):
        """Trích xuất entities từ context bằng OpenAI GPT-4o-mini"""
        if not self.openai_client:
            print("OpenAI client chưa được khởi tạo. Không thể trích xuất entities.")
            return []
        
        try:
            # Prompt để trích xuất entities bao gồm ngày tháng và số lượng quan trọng
            prompt = f"""
Bạn là một chuyên gia trích xuất thông tin cho hệ thống fact-checking. Hãy trích xuất tất cả các thực thể quan trọng từ văn bản sau, bao gồm CẢ NGÀY THÁNG và SỐ LƯỢNG QUAN TRỌNG.
Quan trọng, chỉ lấy những từ có trong văn bản, không lấy những từ không có trong văn bản. Nếu trích xuất được các từ thì phải để nó giống y như trong văn bản không được thay đổi.

NGUYÊN TẮC TRÍCH XUẤT:
- Lấy TÊN THỰC THỂ THUẦN TÚY + NGÀY THÁNG + SỐ LƯỢNG QUAN TRỌNG
- Loại bỏ từ phân loại không cần thiết: "con", "chiếc", "cái", "người" (trừ khi là phần của tên riêng)
- Giữ nguyên số đo lường có ý nghĩa thực tế
YÊU CẦU:
Chỉ lấy những từ/cụm từ xuất hiện trong văn bản, giữ nguyên chính tả, không tự thêm hoặc sửa đổi.
Với mỗi thực thể, chỉ lấy một lần (không lặp lại), kể cả xuất hiện nhiều lần trong văn bản.
Nếu thực thể là một phần của cụm danh từ lớn hơn (ví dụ: "đoàn cứu hộ Việt Nam"), hãy trích xuất cả cụm danh từ lớn ("đoàn cứu hộ Việt Nam") và thực thể nhỏ bên trong ("Việt Nam").
Không bỏ sót thực thể chỉ vì nó nằm trong cụm từ khác hoặc là một phần của tên dài.

Các loại thực thể CẦN trích xuất:
1. **Tên loài/sinh vật**: "Patagotitan mayorum", "titanosaur", "voi châu Phi"
2. **Địa danh**: "Argentina", "London", "Neuquen", "TP.HCM", "Quận 6"
3. **Địa danh kết hợp**: "Bảo tàng Lịch sử tự nhiên London", "Nhà máy nước Tân Hiệp"
4. **Tên riêng người**: "Nguyễn Văn A", "Phạm Văn Chính", "Sinead Marron"
5. **Tổ chức**: "Bảo tàng Lịch sử tự nhiên", "SAWACO", "Microsoft", "PLO"
6. **Sản phẩm/công nghệ**: "iPhone", "ChatGPT", "PhoBERT", "dịch vụ cấp nước"

7. **NGÀY THÁNG & THỜI GIAN QUAN TRỌNG**:
   - Năm: "2010", "2017", "2022"
   - Ngày tháng: "25-3", "15/4/2023", "ngày 10 tháng 5"
   - Giờ cụ thể: "22 giờ", "6h30", "14:30"
   - Khoảng thời gian: "từ 22 giờ đến 6 giờ", "2-3 ngày"

8. **SỐ LƯỢNG & ĐO LƯỜNG QUAN TRỌNG**:
   - Kích thước vật lý: "37m", "69 tấn", "6m", "180cm"
   - Số lượng có ý nghĩa: "6 con", "12 con", "100 người"  
   - Giá trị tiền tệ: "5 triệu đồng", "$100", "€50"
   - Tỷ lệ phần trăm: "80%", "15%"
   - Nhiệt độ: "25°C", "100 độ"

KHÔNG lấy (số lượng không có ý nghĩa):
- Số thứ tự đơn lẻ: "1", "2", "3" (trừ khi là năm hoặc địa chỉ)
- Từ chỉ số lượng mơ hồ: "nhiều", "ít", "vài", "một số"
- Đơn vị đo đơn lẻ: "mét", "tấn", "kg" (phải có số đi kèm)

Ví dụ INPUT: "6 con titanosaur ở Argentina nặng 69 tấn, được trưng bày tại Bảo tàng Lịch sử tự nhiên London từ năm 2017 lúc 14:30"
Ví dụ OUTPUT: ["titanosaur", "Argentina", "69 tấn", "Bảo tàng Lịch sử tự nhiên London", "2017", "14:30", "6 con"]

Ví dụ INPUT: "SAWACO thông báo cúp nước tại Quận 6 từ 22 giờ ngày 25-3 đến 6 giờ ngày 26-3"
Ví dụ OUTPUT: ["SAWACO", "Quận 6", "22 giờ", "25-3", "6 giờ", "26-3"]

Trả về JSON array: ["entity1", "entity2", "entity3"]

Văn bản:
{context_text}
"""

            # Use parameters from CLI if available
            model = getattr(self, 'openai_model', 'gpt-4o-mini')
            temperature = getattr(self, 'openai_temperature', 0.0)
            max_tokens = getattr(self, 'openai_max_tokens', 1000)

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=max_tokens
            )
            
            # Parse response
            response_content = response.choices[0].message.content
            if response_content is None:
                print("OpenAI API trả về content None")
                return []
            response_text = response_content.strip()
            
            # Strip markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove '```json'
            if response_text.startswith('```'):
                response_text = response_text[3:]   # Remove '```'
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove ending '```'
            response_text = response_text.strip()
            
            # Cố gắng parse JSON
            try:
                entities = json.loads(response_text)
                if isinstance(entities, list):
                    # Filter out empty strings and duplicates
                    entities = list(set([entity.strip() for entity in entities if entity.strip()]))
                    return entities
                else:
                    print(f"Response không phải dạng list: {response_text}")
                    return []
            except json.JSONDecodeError:
                print(f"Không thể parse JSON từ OpenAI response: {response_text}")
                return []
                
        except Exception as e:
            print(f"Lỗi khi gọi OpenAI API: {e}")
            return []
    
    def normalize_text(self, text):
        if not text:
            return ""
        # Loại bỏ dấu câu, chuyển về lower, loại bỏ dấu tiếng Việt
        text = text.lower()
        text = re.sub(r'[\W_]+', ' ', text)  # bỏ ký tự không phải chữ/số
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fuzzy_in(self, entity, claim_text, threshold=0.8):
        # So sánh fuzzy: entity có xuất hiện gần đúng trong claim_text không
        if entity in claim_text:
            return True
        # Nếu entity là cụm từ, kiểm tra từng từ
        for word in entity.split():
            if word in claim_text:
                return True
        # Fuzzy match toàn chuỗi
        ratio = SequenceMatcher(None, entity, claim_text).ratio()
        return ratio >= threshold

    def improved_entity_matching(self, entity, sentence_text, model=None):
        """
        ✅ ENHANCED: Improved entity matching với better underscore handling
        """
        # Normalize both texts để consistent comparison
        entity_normalized = self.normalize_text(entity)
        sentence_normalized = self.normalize_text(sentence_text)
        
        # Method 1: Normalized direct matching (most reliable)
        if entity_normalized in sentence_normalized:
            return True
        
        # Method 2: Original case-insensitive matching (fallback)
        entity_lower = entity.lower()
        sentence_lower = sentence_text.lower()
        if entity_lower in sentence_lower:
            return True
            
        # Method 3: Bidirectional underscore/space variants
        entity_variants = [
            entity.replace(" ", "_").lower(),  # space -> underscore
            entity.replace("_", " ").lower(),  # underscore -> space
            entity_lower
        ]
        sentence_variants = [
            sentence_text.replace(" ", "_").lower(),
            sentence_text.replace("_", " ").lower(), 
            sentence_lower
        ]
        
        # Cross-match all variants
        for e_variant in entity_variants:
            for s_variant in sentence_variants:
                if e_variant in s_variant:
                    return True
        
        # Method 4: VnCoreNLP segmentation (if available)
        if model and segment_entity_with_vncorenlp:
            try:
                entity_vncorenlp_seg = segment_entity_with_vncorenlp(entity, model).lower()
                if entity_vncorenlp_seg in sentence_lower:
                    return True
            except:
                pass
        
        # Method 5: Enhanced word-level matching với underscore handling
        entity_words = entity_normalized.split()
        if len(entity_words) > 1:
            all_words_found = True
            for word in entity_words:
                if len(word) <= 2:  # Skip very short words
                    continue
                    
                # Create word variants
                word_variants = [
                    word,
                    word.replace(" ", "_"),
                    word.replace("_", " ")
                ]
                
                # Check if any variant exists in normalized sentence
                word_found = any(variant in sentence_normalized for variant in word_variants if variant)
                if not word_found:
                    all_words_found = False
                    break
                    
            if all_words_found:
                return True
        
        return False

    def _check_entity_word_overlap(self, entity):
        """
        ✅ NEW: Check if entity overlaps with existing word nodes để tránh duplicates
        """
        entity_normalized = self.normalize_text(entity)
        overlapping_words = []
        
        for word, word_node_id in self.word_nodes.items():
            word_normalized = self.normalize_text(word)
            
            # Check exact match hoặc substring overlap
            if (entity_normalized == word_normalized or 
                entity_normalized in word_normalized or 
                word_normalized in entity_normalized):
                overlapping_words.append((word, word_node_id))
        
        return overlapping_words

    def add_entities_to_graph(self, entities, context_sentences, model=None, entity_scores=None):
        """
        ✅ ENHANCED: Thêm entities vào graph với duplicate detection và enhanced scoring
        
        Args:
            entities: List of entity strings  
            context_sentences: Context sentences
            model: Optional model for entity matching
            entity_scores: Dict mapping entity -> score (for dual-source entities)
        """
        entity_nodes_added = []
        total_connections = 0
        
        # Lấy claim text (nếu có claim node)
        claim_text = None
        if hasattr(self, 'claim_node') and self.claim_node and self.claim_node in self.graph.nodes:
            claim_text = self.graph.nodes[self.claim_node]['text']
            claim_text_norm = self.normalize_text(claim_text)
        else:
            claim_text_norm = None
            
        # Track dual-source entity performance & duplicates
        dual_source_connections = 0
        single_source_connections = 0
        duplicate_avoidance_count = 0
        
        for entity in entities:
            # Get entity score (default 1.0 if not provided)
            entity_score = 1.0
            if entity_scores:
                entity_normalized = entity.lower().strip()
                for norm_entity, score_info in entity_scores.items():
                    if (norm_entity == entity_normalized or 
                        entity_normalized in norm_entity or 
                        norm_entity in entity_normalized):
                        entity_score = score_info.get('score', 1.0)
                        break
            
            # ✅ NEW: Check for overlapping word nodes trước khi tạo entity node
            overlapping_words = self._check_entity_word_overlap(entity)
            
            if overlapping_words:
                # Use existing word node và enhance it với entity properties
                word_text, existing_word_node = overlapping_words[0]  # Take first match
                entity_node = existing_word_node
                
                # ✅ Enhance existing word node với entity attributes
                self.graph.nodes[entity_node]['entity_score'] = entity_score
                self.graph.nodes[entity_node]['is_dual_source'] = entity_score >= 2.0
                self.graph.nodes[entity_node]['is_enhanced_word'] = True  # Mark as enhanced
                self.graph.nodes[entity_node]['original_entity'] = entity
                
                duplicate_avoidance_count += 1
                # print(f"🔄 Reusing word node '{word_text}' for entity '{entity}' (score={entity_score:.1f})")
            else:
                # Tạo entity node mới với score
                entity_node = self.add_entity_node(entity, entity_type="ENTITY")
                # ✅ Store entity score in node attributes
                self.graph.nodes[entity_node]['entity_score'] = entity_score
                self.graph.nodes[entity_node]['is_dual_source'] = entity_score >= 2.0
            
            entity_nodes_added.append(entity_node)
            entity_connections = 0
            
            # Tìm các sentences có chứa entity này
            for sent_idx, sentence_node in self.sentence_nodes.items():
                sentence_text = self.graph.nodes[sentence_node]['text']
                if self.improved_entity_matching(entity, sentence_text, model):
                    # ✅ Apply weighted connections based on entity score
                    connection_weight = entity_score * 1.0  # Base weight * entity score
                    self.graph.add_edge(entity_node, sentence_node, 
                                      relation="contains", 
                                      edge_type="entity_structural",
                                      weight=connection_weight,
                                      entity_score=entity_score)
                    entity_connections += 1
                    total_connections += 1
                    
                    # Track dual-source performance
                    if entity_score >= 2.0:
                        dual_source_connections += 1
                    else:
                        single_source_connections += 1
            
            # Kết nối entity với claim nếu entity xuất hiện trong claim
            if claim_text_norm:
                entity_norm = self.normalize_text(entity)
                if self.fuzzy_in(entity_norm, claim_text_norm, threshold=0.8):
                    # ✅ Apply higher weight for dual-source entities in claim connections
                    claim_connection_weight = entity_score * 1.5  # Extra boost for claim connections
                    self.graph.add_edge(entity_node, self.claim_node, 
                                      relation="mentioned_in", 
                                      edge_type="entity_structural",
                                      weight=claim_connection_weight,
                                      entity_score=entity_score)
            
            if entity_connections == 0:
                connection_status = "dual-source" if entity_score >= 2.0 else "single-source"
                # print(f"⚠️ Entity '{entity}' ({connection_status}, score={entity_score:.1f}) không kết nối với sentence nào")
        
        # Enhanced logging with duplicate avoidance (minimal)
        # print(f"📊 Entity Graph Enhancement:")
        # print(f"   🎯 Dual-source connections: {dual_source_connections}")
        # print(f"   📝 Single-source connections: {single_source_connections}")
        # print(f"   🔄 Duplicate nodes avoided: {duplicate_avoidance_count}")
        # print(f"   📈 Total weighted connections: {total_connections}")
        # print(f"   ✅ Efficiency: {len(entity_nodes_added)} nodes created for {len(entities)} entities")
        
        return entity_nodes_added
    
    def extract_and_add_entities(self, context_text, context_sentences):
        """Phương thức chính để trích xuất và thêm entities vào graph"""
        entities = self.extract_entities_with_openai(context_text)
        
        if entities:
            entity_nodes = self.add_entities_to_graph(entities, context_sentences)
            return entity_nodes
        else:
            print("Không có entities nào được trích xuất.")
            return []
    
    def _init_phobert_model(self):
        """Khởi tạo PhoBERT model"""
        try:
            self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
        except Exception as e:
            print(f"Lỗi khi khởi tạo PhoBERT model: {e}")
    
    def get_word_embeddings(self, words):
        """Lấy embeddings của các từ"""
        if not self.phobert_tokenizer or not self.phobert_model:
            print("PhoBERT model chưa được khởi tạo. Không thể lấy embeddings.")
            return None
        
        embeddings = []
        for word in words:
            if word not in self.word_embeddings:
                inputs = self.phobert_tokenizer(word, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.phobert_model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
                self.word_embeddings[word] = embeddings[-1]
            else:
                embeddings.append(self.word_embeddings[word])
        
        return np.array(embeddings)
    
    def get_similarity(self, word1, word2):
        if not cosine_similarity:
            print("cosine_similarity không khả dụng.")
            return 0.0
        if word1 not in self.word_embeddings or word2 not in self.word_embeddings:
            print(f"Từ '{word1}' hoặc '{word2}' không có trong word_embeddings.")
            return 0.0
        embedding1 = self.word_embeddings[word1]
        embedding2 = self.word_embeddings[word2]
        return cosine_similarity([embedding1], [embedding2])[0][0]
    
    def get_similar_words(self, word, top_k=5):
        """Tìm các từ có độ tương đồng cao với từ đã cho"""
        if word not in self.word_embeddings:
            return []
        
        similarities = []
        for other_word in self.word_embeddings.keys():
            if other_word != word:
                similarity = self.get_similarity(word, other_word)
                similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [word for word, similarity in similarities[:top_k]]
    
    def get_sentence_embeddings(self, sentences):
        """Lấy embeddings của các câu"""
        if not self.phobert_tokenizer or not self.phobert_model:
            print("PhoBERT model chưa được khởi tạo. Không thể lấy embeddings.")
            return None
        
        embeddings = []
        for sentence in sentences:
            inputs = self.phobert_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = self.phobert_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        
        return np.array(embeddings)
    
    def get_sentence_similarity(self, sentence1, sentence2):
        """Tính độ tương đồng giữa hai câu"""
        # Lấy embeddings cho cả 2 câu
        embeddings = self.get_sentence_embeddings([sentence1, sentence2])
        if embeddings is None or len(embeddings) < 2:
            return 0.0
        
        if cosine_similarity is None:
            print("cosine_similarity không khả dụng - sử dụng numpy dot product")
            # Tính cosine similarity bằng numpy
            emb1_norm = embeddings[0] / np.linalg.norm(embeddings[0])
            emb2_norm = embeddings[1] / np.linalg.norm(embeddings[1])
            return np.dot(emb1_norm, emb2_norm)
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    def build_semantic_similarity_edges(self, use_faiss=True):
        """Xây dựng các cạnh semantic similarity giữa các từ (không sử dụng PCA)"""
        
        # Lấy tất cả word nodes
        word_nodes = [node_id for node_id in self.graph.nodes() 
                     if self.graph.nodes[node_id]['type'] == 'word']
        
        if len(word_nodes) < 2:
            print("Cần ít nhất 2 word nodes để xây dựng semantic edges.")
            return
        
        # Lấy danh sách từ và POS tags
        words = []
        pos_tags = []
        word_node_mapping = {}
        
        for node_id in word_nodes:
            word = self.graph.nodes[node_id]['text']
            pos = self.graph.nodes[node_id].get('pos', '')
            words.append(word)
            pos_tags.append(pos)
            word_node_mapping[word] = node_id
        
        # Lấy embeddings (sử dụng full PhoBERT embeddings - không PCA)
        embeddings = self.get_word_embeddings(words)
        if embeddings is None:
            print("Không thể lấy embeddings.")
            return
        
        # Xây dựng Faiss index (optional)
        if use_faiss:
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (for cosine similarity)
            
            # Normalize vectors for cosine similarity
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.faiss_index.add(embeddings_normalized.astype(np.float32))
            
            # Create mappings
            self.word_to_index = {word: i for i, word in enumerate(words)}
            self.index_to_word = {i: word for i, word in enumerate(words)}
        else:
            # Normalize embeddings để tính cosine similarity nhanh hơn
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Tìm similar words và tạo edges
        edges_added = 0
        
        for i, word1 in enumerate(words):
            pos1 = pos_tags[i]
            node1 = word_node_mapping[word1]
            
            if use_faiss and self.faiss_index is not None:
                # Sử dụng Faiss để tìm similar words
                query_vector = embeddings_normalized[i:i+1].astype(np.float32)
                similarities, indices = self.faiss_index.search(query_vector, self.top_k_similar + 1)  # +1 vì sẽ bao gồm chính nó
                
                for j, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx == i:  # Skip chính nó
                        continue
                    
                    if similarity < self.similarity_threshold:
                        continue
                    
                    word2 = self.index_to_word[idx]
                    pos2 = pos_tags[idx]
                    node2 = word_node_mapping[word2]
                    
                    # Chỉ kết nối từ cùng loại POS (optional)
                    if pos1 and pos2 and pos1 == pos2:
                        if not self.graph.has_edge(node1, node2):
                            self.graph.add_edge(node1, node2, 
                                              relation="semantic_similar", 
                                              edge_type="semantic",
                                              similarity=float(similarity))
                            edges_added += 1
            else:
                # Sử dụng numpy matrix multiplication (nhanh hơn sklearn cho cosine similarity)
                for j, word2 in enumerate(words):
                    if i >= j:  # Tránh duplicate và self-comparison
                        continue
                    
                    pos2 = pos_tags[j]
                    
                    # Chỉ so sánh từ cùng loại POS
                    if pos1 and pos2 and pos1 != pos2:
                        continue
                    
                    # Tính cosine similarity với normalized vectors (nhanh hơn)
                    similarity = np.dot(embeddings_normalized[i], embeddings_normalized[j])
                    
                    if similarity >= self.similarity_threshold:
                        node2 = word_node_mapping[word2]
                        if not self.graph.has_edge(node1, node2):
                            self.graph.add_edge(node1, node2, 
                                              relation="semantic_similar", 
                                              edge_type="semantic",
                                              similarity=float(similarity))
                            edges_added += 1
        
        return edges_added
    
    def get_semantic_statistics(self):
        """Thống kê về semantic edges"""
        semantic_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'semantic'
        ]
        
        if not semantic_edges:
            return {
                "total_semantic_edges": 0,
                "average_similarity": 0.0,
                "similarity_distribution": {}
            }
        
        similarities = [data.get('similarity', 0.0) for u, v, data in semantic_edges]
        
        return {
            "total_semantic_edges": len(semantic_edges),
            "average_similarity": np.mean(similarities),
            "max_similarity": np.max(similarities),
            "min_similarity": np.min(similarities),
            "similarity_distribution": {
                "0.85-0.90": len([s for s in similarities if 0.85 <= s < 0.90]),
                "0.90-0.95": len([s for s in similarities if 0.90 <= s < 0.95]),
                "0.95-1.00": len([s for s in similarities if 0.95 <= s <= 1.00])
            }
        }
    
    def beam_search_paths(self, beam_width=10, max_depth=6, max_paths=20):
        """
        Tìm đường đi từ claim đến sentence nodes bằng Beam Search
        
        Args:
            beam_width (int): Độ rộng beam search
            max_depth (int): Độ sâu tối đa của path
            max_paths (int): Số lượng paths tối đa trả về
            
        Returns:
            List[Path]: Danh sách paths tốt nhất
        """
        if not self.claim_node:
            print("⚠️ Không có claim node để thực hiện beam search")
            return []
            
        # Tạo BeamSearchPathFinder
        path_finder = BeamSearchPathFinder(
            text_graph=self,
            beam_width=beam_width,
            max_depth=max_depth
        )
        
        # Tìm paths
        paths = path_finder.find_best_paths(max_paths=max_paths)
        
        return paths
    
    def export_beam_search_results(self, paths, output_dir="output", file_prefix="beam_search"):
        """
        Export kết quả beam search ra files
        
        Args:
            paths: Danh sách paths từ beam search
            output_dir (str): Thư mục output
            file_prefix (str): Prefix cho tên file
            
        Returns:
            tuple: (json_file_path, summary_file_path)
        """
        if not paths:
            print("⚠️ Không có paths để export")
            return None, None
            
        # Tạo BeamSearchPathFinder để export
        path_finder = BeamSearchPathFinder(self)
        
        # Export JSON và summary với absolute paths
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure we use the correct directory
        current_dir = os.getcwd()
        if current_dir.endswith('vncorenlp'):
            # If we're in vncorenlp directory, go back to parent
            current_dir = os.path.dirname(current_dir)
        
        json_file = os.path.join(current_dir, output_dir, f"{file_prefix}_{timestamp}.json")
        summary_file = os.path.join(current_dir, output_dir, f"{file_prefix}_summary_{timestamp}.txt")
        
        json_path = path_finder.export_paths_to_file(paths, json_file)
        summary_path = path_finder.export_paths_summary(paths, summary_file)
        
        return json_path, summary_path
    
    def analyze_paths_quality(self, paths):
        """
        Phân tích chất lượng của các paths tìm được
        
        Args:
            paths: Danh sách paths
            
        Returns:
            dict: Thống kê về paths
        """
        if not paths:
            return {
                'total_paths': 0,
                'avg_score': 0,
                'avg_length': 0,
                'paths_to_sentences': 0,
                'paths_through_entities': 0
            }
            
        total_paths = len(paths)
        scores = [p.score for p in paths]
        lengths = [len(p.nodes) for p in paths]
        
        sentences_reached = sum(1 for p in paths if any(
            node.startswith('sentence') for node in p.nodes
        ))
        
        entities_visited = sum(1 for p in paths if p.entities_visited)
        
        return {
            'total_paths': total_paths,
            'avg_score': sum(scores) / total_paths if scores else 0,
            'max_score': max(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'avg_length': sum(lengths) / total_paths if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'paths_to_sentences': sentences_reached,
            'paths_through_entities': entities_visited,
            'sentence_reach_rate': sentences_reached / total_paths if total_paths > 0 else 0,
            'entity_visit_rate': entities_visited / total_paths if total_paths > 0 else 0
        }
    
    def multi_level_beam_search_paths(
        self,
        max_levels: int = 3,
        beam_width_per_level: int = 3,
        max_depth: int = 30,
        allow_skip_edge: bool = False,        # 🆕 bật/tắt 2-hops
        min_new_sentences: int = 0,            # đã có từ lần trước
        advanced_data_filter=None,
        claim_text="",
        entities=None,
        filter_top_k: int = 5
    ) -> Dict[int, List]:
        """
        Multi-level beam search wrapper cho TextGraph
        
        Args:
            max_levels: Số levels tối đa
            beam_width_per_level: Số sentences mỗi level
            max_depth: Độ sâu tối đa cho beam search
            
        Returns:
            Dict[level, List[Path]]: Results theo từng level
        """
        if not self.claim_node:
            print("⚠️ Không có claim node để thực hiện multi-level beam search")
            return {}
            
        # Tạo BeamSearchPathFinder với custom max_depth
        path_finder = BeamSearchPathFinder(
            text_graph=self,
            beam_width=25,
            max_depth=max_depth,
            allow_skip_edge=allow_skip_edge    # 🆕 chuyển tham số
        )
        
        # Chạy multi-level search
        multi_results = path_finder.multi_level_beam_search(
            max_levels=max_levels,
            beam_width_per_level=beam_width_per_level,
            min_new_sentences=min_new_sentences,
            sbert_model=getattr(self, 'sbert_model', None),
            claim_text=claim_text,
            entities=entities,
            filter_top_k=filter_top_k,
            use_phobert=False
        )
        
        return multi_results 

    def build_sentence_claim_semantic_edges(self, similarity_threshold=0.7):
        """
        Tạo kết nối trực tiếp giữa sentences và claim dựa trên độ tương đồng ngữ nghĩa
        sử dụng PhoBERT embeddings
        
        Args:
            similarity_threshold (float): Ngưỡng độ tương đồng để tạo kết nối (0.0-1.0)
        """
        if not self.claim_node or not self.phobert_model:
            print("⚠️ Cần có claim node và PhoBERT model để tạo sentence-claim semantic edges")
            return 0
        
        # Lấy claim text
        claim_text = self.graph.nodes[self.claim_node]['text']
        
        # Lấy tất cả sentence nodes
        sentence_nodes = [(node_id, self.graph.nodes[node_id]['text']) 
                         for node_id in self.graph.nodes() 
                         if self.graph.nodes[node_id]['type'] == 'sentence']
        
        if not sentence_nodes:
            print("⚠️ Không có sentence nodes để so sánh với claim")
            return 0
        
        edges_added = 0
        
        # So sánh từng sentence với claim
        for sentence_node_id, sentence_text in sentence_nodes:
            # Tính độ tương đồng
            similarity = self.get_sentence_similarity(sentence_text, claim_text)
            
            # Nếu đạt ngưỡng thì tạo edge
            if similarity >= similarity_threshold:
                # Kiểm tra xem edge đã tồn tại chưa
                existing_edge = False
                if self.graph.has_edge(sentence_node_id, self.claim_node):
                    # Kiểm tra xem có edge semantic nào chưa
                    edge_data = self.graph.edges[sentence_node_id, self.claim_node]
                    if edge_data.get('edge_type') == 'sentence_claim_semantic':
                        existing_edge = True
                
                if not existing_edge:
                    self.graph.add_edge(
                        sentence_node_id, 
                        self.claim_node,
                        relation="semantically_similar",
                        edge_type="sentence_claim_semantic",
                        similarity=float(similarity)
                    )
                    edges_added += 1
        
        return edges_added
    
    def get_sentence_claim_semantic_statistics(self):
        """Thống kê về các kết nối semantic giữa sentence và claim"""
        semantic_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'sentence_claim_semantic'
        ]
        
        if not semantic_edges:
            return {
                "total_sentence_claim_edges": 0,
                "average_similarity": 0.0,
                "connected_sentences": []
            }
        
        similarities = [data.get('similarity', 0.0) for u, v, data in semantic_edges]
        connected_sentences = []
        
        for u, v, data in semantic_edges:
            # u là sentence, v là claim (hoặc ngược lại)
            sentence_node = u if self.graph.nodes[u]['type'] == 'sentence' else v
            sentence_text = self.graph.nodes[sentence_node]['text'][:100] + "..." if len(self.graph.nodes[sentence_node]['text']) > 100 else self.graph.nodes[sentence_node]['text']
            
            connected_sentences.append({
                'sentence_id': sentence_node,
                'sentence_text': sentence_text,
                'similarity': data.get('similarity', 0.0)
            })
        
        # Sắp xếp theo độ tương đồng giảm dần
        connected_sentences.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            "total_sentence_claim_edges": len(semantic_edges),
            "average_similarity": np.mean(similarities) if similarities else 0.0,
            "max_similarity": np.max(similarities) if similarities else 0.0,
            "min_similarity": np.min(similarities) if similarities else 0.0,
            "connected_sentences": connected_sentences,
            "similarity_distribution": {
                "0.75-0.80": len([s for s in similarities if 0.75 <= s < 0.80]),
                "0.80-0.85": len([s for s in similarities if 0.80 <= s < 0.85]),
                "0.85-0.90": len([s for s in similarities if 0.85 <= s < 0.90]),
                "0.90-1.00": len([s for s in similarities if 0.90 <= s <= 1.00])
            }
        } 

    def get_direct_connected_sentences(self):
        """
        Lấy danh sách sentences được kết nối trực tiếp với claim 
        thông qua sentence-claim semantic edges
        
        Returns:
            List[Dict]: List of {'sentence_id': str, 'sentence_text': str, 'similarity': float}
        """
        if not self.claim_node:
            return []
        
        direct_sentences = []
        
        # Tìm tất cả sentence-claim semantic edges
        for neighbor in self.graph.neighbors(self.claim_node):
            edge_data = self.graph.edges[neighbor, self.claim_node]
            
            # Chỉ lấy sentence-claim semantic edges
            if edge_data.get('edge_type') == 'sentence_claim_semantic':
                neighbor_data = self.graph.nodes[neighbor]
                if neighbor_data.get('type') == 'sentence':
                    direct_sentences.append({
                        'sentence_id': neighbor,
                        'sentence_text': neighbor_data.get('text', ''),
                        'similarity': edge_data.get('similarity', 0.0)
                    })
        
        # Sắp xếp theo độ tương đồng giảm dần
        direct_sentences.sort(key=lambda x: x['similarity'], reverse=True)
        
        return direct_sentences
    
    def enhanced_multi_level_beam_search_with_direct_connections(
        self,
        max_levels: int = 3,
        beam_width_per_level: int = 3,
        max_depth: int = 30,
        min_new_sentences: int = 2,
        advanced_data_filter=None,
        claim_text="",
        entities=None,
        filter_top_k: int = 5,
        use_direct_as_starting_points: bool = True,
        sort_by_original_order: bool = False
    ) -> Dict:
        """
        Enhanced multi-level beam search sử dụng sentence-claim semantic edges
        
        Workflow:
        1. Tạo sentence-claim semantic edges (nếu chưa có)
        2. Lấy direct connected sentences làm starting points
        3. Beam search từ direct sentences qua các levels
        4. Merge direct + multi-hop results 
        5. Final reranking (sẵn sàng cho SBERT)
        
        Args:
            use_direct_as_starting_points: Dùng direct sentences làm starting points
            
        Returns:
            Dict: {
                'direct_connections': List[Dict],
                'multi_hop_results': Dict[int, List[Path]], 
                'merged_sentences': List[Dict],
                'statistics': Dict
            }
        """
        # print("🚀 ENHANCED MULTI-LEVEL BEAM SEARCH WITH DIRECT CONNECTIONS")
        # print("=" * 65)
        
        # Step 1: Đảm bảo có sentence-claim semantic edges
        # print("🔗 Step 1: Kiểm tra sentence-claim semantic edges...")
        semantic_stats = self.get_sentence_claim_semantic_statistics()
        
        if semantic_stats['total_sentence_claim_edges'] == 0:
            # print("   ⚠️ Chưa có sentence-claim semantic edges. Tạo mới với threshold 0.75...")
            edges_added = self.build_sentence_claim_semantic_edges(similarity_threshold=0.7)
            # print(f"   ✅ Đã tạo {edges_added} sentence-claim semantic edges")
        else:
            # print(f"   ✅ Đã có {semantic_stats['total_sentence_claim_edges']} sentence-claim semantic edges")
            pass
        
        # Step 2: Lấy direct connected sentences  
        # print("\n📋 Step 2: Lấy sentences kết nối trực tiếp với claim...")
        direct_sentences = self.get_direct_connected_sentences()
        # print(f"   Tìm được {len(direct_sentences)} direct connected sentences")
        
        # for i, sent in enumerate(direct_sentences[:3]):  # Show top 3
        #     print(f"   {i+1}. [{sent['similarity']:.3f}] {sent['sentence_text'][:80]}...")
        
        # Step 3: Modified beam search sử dụng direct sentences làm starting points
        # print(f"\n🎯 Step 3: Multi-level beam search từ direct sentences...")
        
        if use_direct_as_starting_points and direct_sentences:
            # Beam search từ direct sentences thay vì từ claim
            multi_hop_results = self._beam_search_from_direct_sentences(
                direct_sentences=direct_sentences,
                max_levels=max_levels,
                beam_width_per_level=beam_width_per_level,
                max_depth=max_depth,
                min_new_sentences=min_new_sentences,
                claim_text=claim_text,
                entities=entities,
                filter_top_k=filter_top_k
            )
        else:
            # Fallback to traditional beam search từ claim
            # print("   📌 Fallback: Traditional beam search từ claim...")
            multi_hop_results = self.multi_level_beam_search_paths(
                max_levels=max_levels,
                beam_width_per_level=beam_width_per_level,
                max_depth=max_depth,
                min_new_sentences=min_new_sentences,
                claim_text=claim_text,
                entities=entities,
                filter_top_k=filter_top_k
            )
        
        # Step 4: Merge direct + multi-hop results
        # print(f"\n🔄 Step 4: Merge direct connections với multi-hop results...")
        merged_sentences = self._merge_direct_and_multihop_results(
            direct_sentences=direct_sentences,
            multi_hop_results=multi_hop_results,
            sort_by_original_order=sort_by_original_order
        )
        
        # print(f"   Merged total: {len(merged_sentences)} unique sentences")
        
        # Step 5: Thống kê và chuẩn bị cho final reranking
        # print(f"\n📊 Step 5: Chuẩn bị kết quả cho final SBERT reranking...")
        
        statistics = {
            'direct_sentences_count': len(direct_sentences),
            'multi_hop_levels': len(multi_hop_results),
            'total_multi_hop_sentences': sum(
                len(self._extract_sentences_from_paths(paths)) 
                for paths in multi_hop_results.values()
            ) if multi_hop_results else 0,
            'merged_sentences_count': len(merged_sentences),
            'semantic_edges_count': semantic_stats['total_sentence_claim_edges']
        }
        
        # for key, value in statistics.items():
        #     print(f"   • {key}: {value}")
        
        # print(f"\n✅ READY FOR FINAL SBERT RERANKING!")
        # print(f"   Merged sentences can be passed to SBERT reranker")
        
        return {
            'direct_connections': direct_sentences,
            'multi_hop_results': multi_hop_results,
            'merged_sentences': merged_sentences,
            'statistics': statistics
        }
    
    def _beam_search_from_direct_sentences(
        self,
        direct_sentences: List[Dict],
        max_levels: int = 3,
        beam_width_per_level: int = 3,
        max_depth: int = 30,
        min_new_sentences: int = 2,
        claim_text="",
        entities=None,
        filter_top_k: int = 5
    ) -> Dict[int, List]:
        """
        Beam search từ direct connected sentences thay vì từ claim
        
        Args:
            direct_sentences: List sentences kết nối trực tiếp với claim
            
        Returns:
            Dict[level, List[Path]]: Multi-hop results by level
        """
        from .beam_search import BeamSearchPathFinder, Path
        
        # Tạo BeamSearchPathFinder
        path_finder = BeamSearchPathFinder(
            text_graph=self,
            beam_width=25,
            max_depth=max_depth
        )
        
        results = {}
        all_found_sentences = set()
        
        # Level 0: Direct sentences (đã có sẵn)
        print(f"   📍 LEVEL 0: Direct connected sentences ({len(direct_sentences)} sentences)")
        direct_sentence_ids = [s['sentence_id'] for s in direct_sentences]
        all_found_sentences.update(direct_sentence_ids)
        
        # Tạo "fake paths" cho direct sentences để có cùng format
        level_0_paths = []
        for sent in direct_sentences:
            # Path từ claim → sentence (direct connection)
            edge_info = (self.claim_node, sent['sentence_id'], 'sentence_claim_semantic')
            path = Path(
                nodes=[self.claim_node, sent['sentence_id']],
                edges=[edge_info],
                score=sent['similarity'] * 5.0  # Boost direct connections
            )
            path.word_matches = set()
            path.path_words = set()
            path.entities_visited = set()
            level_0_paths.append(path)
        
        results[0] = level_0_paths
        current_sentence_nodes = direct_sentence_ids
        
        # Levels 1 to k: Beam search từ direct sentences
        for level in range(1, max_levels + 1):
            if not current_sentence_nodes:
                print(f"     No sentences to expand from level {level-1}")
                break
                
            print(f"   📍 LEVEL {level}: Từ {len(current_sentence_nodes)} sentences → New sentences")
            level_paths = []
            new_sentence_nodes = set()
            
            # Beam search từ mỗi sentence của level trước
            for sentence_node in current_sentence_nodes:
                sentence_paths = path_finder._beam_search_from_sentence(
                    sentence_node, 
                    max_paths=beam_width_per_level,
                    exclude_sentences=all_found_sentences
                )
                
                # Lấy sentences mới
                new_sentences = path_finder._extract_sentence_nodes_from_paths(sentence_paths)
                new_sentences = [s for s in new_sentences if s not in all_found_sentences]
                
                level_paths.extend(sentence_paths)
                new_sentence_nodes.update(new_sentences)
            
            # Giữ lại top sentences
            if level_paths:
                level_paths.sort(key=lambda p: p.score, reverse=True)
                level_paths = level_paths[:beam_width_per_level]
                
                # Lấy sentences mới từ paths
                final_new_sentences = path_finder._extract_sentence_nodes_from_paths(level_paths)
                unique_new = [s for s in final_new_sentences if s not in all_found_sentences]
                
                # 🔍 Apply SBERT/PhoBERT filtering at each level
                if hasattr(self, 'sbert_model') and self.sbert_model and claim_text and unique_new:
                    try:
                        # Get sentence texts from nodes
                        sentence_texts = []
                        for node in unique_new:
                            node_text = self.graph.nodes[node].get("text", "")
                            if node_text:
                                sentence_texts.append(node_text)
                        
                        if sentence_texts:
                            # Check if PhoBERT should be used for level filtering
                            use_phobert_level = getattr(self, 'use_phobert_level_filtering', False)
                            
                            if use_phobert_level and hasattr(self, 'get_sentence_similarity'):
                                # Use PhoBERT for level filtering
                                print(f"     🔍 Using PhoBERT for level {level} filtering...")
                                similarities = []
                                for sent_text in sentence_texts:
                                    try:
                                        sim = self.get_sentence_similarity(sent_text, claim_text)
                                        similarities.append(sim)
                                    except Exception as e:
                                        print(f"     ⚠️ PhoBERT similarity error: {e}")
                                        similarities.append(0.0)  # Fallback score
                            else:
                                # Use SBERT for level filtering
                                print(f"     🔍 Using SBERT for level {level} filtering...")
                                from sklearn.metrics.pairwise import cosine_similarity
                                claim_embedding = self.sbert_model.encode([claim_text])
                                sentence_embeddings = self.sbert_model.encode(sentence_texts)
                                similarities = cosine_similarity(claim_embedding, sentence_embeddings)[0]
                            
                            # Filter by similarity threshold (0.4 for level filtering)
                            similarity_threshold = 0.4
                            filtered_nodes = []
                            for i, (node, similarity) in enumerate(zip(unique_new, similarities)):
                                if similarity >= similarity_threshold:
                                    filtered_nodes.append(node)
                            
                            # Keep at least filter_top_k sentences even if below threshold
                            if len(filtered_nodes) < filter_top_k and unique_new:
                                # Sort by similarity and take top filter_top_k
                                node_sim_pairs = list(zip(unique_new, similarities))
                                node_sim_pairs.sort(key=lambda x: x[1], reverse=True)
                                filtered_nodes = [node for node, _ in node_sim_pairs[:filter_top_k]]
                            
                            unique_new = filtered_nodes
                            print(f"     🔍 SBERT level filtering: {len(unique_new)} sentences retained")
                    except Exception as e:
                        print(f"     ⚠️ Level filtering error: {e}")
                        # Continue with all sentences if filtering fails
                else:
                    print(f"     📦 Collected {len(unique_new)} raw sentences at level {level} (no level filtering)")
                
                # Cập nhật tracking
                results[level] = level_paths
                all_found_sentences.update(unique_new)
                current_sentence_nodes = unique_new
                
                print(f"     → Level {level}: {len(unique_new)} new sentences")
            else:
                print(f"     → Level {level}: No new sentences found")
                break
        
        return results
    
    def _merge_direct_and_multihop_results(
        self,
        direct_sentences: List[Dict],
        multi_hop_results: Dict[int, List],
        sort_by_original_order: bool = False
    ) -> List[Dict]:
        """
        Merge direct connections với multi-hop results, loại bỏ trùng lặp
        
        Returns:
            List[Dict]: Merged sentences với metadata để reranking
        """
        merged_sentences = []
        seen_sentence_ids = set()
        seen_sentence_texts = set()  # ✅ Thêm dedup by text content
        
        def normalize_text_for_dedup(text):
            """Normalize text for deduplication"""
            if not text:
                return ""
            # Remove extra spaces, convert to lowercase, strip
            normalized = ' '.join(text.strip().lower().split())
            return normalized
        
        # 1. Thêm direct connections (highest priority)
        for sent in direct_sentences:
            sentence_text = sent['sentence_text']
            normalized_text = normalize_text_for_dedup(sentence_text)
            
            if (sent['sentence_id'] not in seen_sentence_ids and 
                normalized_text not in seen_sentence_texts):
                
                merged_sentences.append({
                    'sentence_id': sent['sentence_id'],
                    'sentence_text': sentence_text,
                    'source': 'direct_connection',
                    'level': 0,
                    'similarity_score': sent['similarity'],
                    'path_score': sent['similarity'] * 5.0,  # Boost direct
                    'hop_distance': 1  # Direct = 1 hop
                })
                seen_sentence_ids.add(sent['sentence_id'])
                seen_sentence_texts.add(normalized_text)  # ✅ Track text content
        
        # 2. Thêm multi-hop results theo levels
        for level, paths in multi_hop_results.items():
            if level == 0:  # Skip level 0 (đã có trong direct)
                continue
                
            for path in paths:
                # Lấy sentence cuối cùng trong path
                for node in reversed(path.nodes):
                    node_data = self.graph.nodes.get(node, {})
                    if node_data.get('type') == 'sentence':
                        sentence_text = node_data.get('text', '')
                        normalized_text = normalize_text_for_dedup(sentence_text)
                        
                        if (node not in seen_sentence_ids and 
                            normalized_text not in seen_sentence_texts):
                            
                            merged_sentences.append({
                                'sentence_id': node,
                                'sentence_text': sentence_text,
                                'source': 'multi_hop',
                                'level': level,
                                'similarity_score': 0.0,  # No direct similarity
                                'path_score': getattr(path, 'score', 0.0),
                                'hop_distance': level + 1,  # Multi-hop distance
                                'path_length': len(path.nodes)
                            })
                            seen_sentence_ids.add(node)
                            seen_sentence_texts.add(normalized_text)  # ✅ Track text content
                            break  # Chỉ lấy 1 sentence cuối per path
        
        def extract_sentence_index(sentence_id):
            """Extract numeric index from sentence_id like 'sentence_5' -> 5"""
            try:
                if isinstance(sentence_id, str) and sentence_id.startswith('sentence_'):
                    return int(sentence_id.replace('sentence_', ''))
                elif isinstance(sentence_id, str) and sentence_id.isdigit():
                    return int(sentence_id)
                else:
                    return 999999  # Put unknown format at end
            except:
                return 999999
        
        # 3. Sắp xếp theo yêu cầu
        if sort_by_original_order:
            # ✅ Sort by original sentence index (sentence_0, sentence_1, sentence_2...)
            merged_sentences.sort(key=lambda x: (
                x['source'] != 'direct_connection',  # Direct first
                extract_sentence_index(x['sentence_id'])  # Then by original order
            ))
        else:
            # Default: Sort by priority and scores
            merged_sentences.sort(key=lambda x: (
                x['source'] != 'direct_connection',  # Direct first
                -x['path_score'],  # Higher score first
                x['hop_distance']  # Shorter distance first
            ))
        
        return merged_sentences
    
    def _extract_sentences_from_paths(self, paths: List) -> List[str]:
        """Helper: Extract sentence IDs từ paths"""
        sentence_ids = set()
        for path in paths:
            for node in path.nodes:
                node_data = self.graph.nodes.get(node, {})
                if node_data.get('type') == 'sentence':
                    sentence_ids.add(node)
        return list(sentence_ids) 

    def extract_phrases_from_claim(self, claim_text, claim_sentences=None):
        """
        Trích xuất các cụm từ/phrases từ claim thay vì từng từ riêng lẻ
        Ví dụ: "cựu tổng thống Mỹ", "Donald Trump"
        
        Args:
            claim_text: Text của claim
            claim_sentences: VnCoreNLP output của claim
            
        Returns:
            List[str]: Danh sách phrases từ claim
        """
        phrases = []
        
        try:
            if claim_sentences:
                # print("🔗 Extracting phrase entities from claim using VnCoreNLP...")
                
                for sent_idx, sentence_tokens in claim_sentences.items():
                    current_phrase = []
                    current_pos_sequence = []
                    
                    # Stop words không nên bắt đầu phrase
                    stop_words = {'là', 'của', 'và', 'có', 'trong', 'để', 'với', 'từ', 'này', 'đó', 'các', 'một', 'những'}
                    
                    for i, token in enumerate(sentence_tokens):
                        word = token.get("wordForm", "").strip()
                        pos_tag = token.get("posTag", "")
                        
                        # Bỏ qua punctuation
                        if pos_tag in ['CH', '.', ',', '!', '?', ';', ':']:
                            # Finish current phrase if exists
                            if current_phrase and len(current_phrase) >= 1:
                                phrase_text = " ".join(current_phrase)
                                if len(phrase_text) > 2 and phrase_text.lower() not in stop_words:
                                    phrases.append(phrase_text)
                                    # print(f"   🔗 Phrase: '{phrase_text}' (POS: {' '.join(current_pos_sequence)})")
                            current_phrase = []
                            current_pos_sequence = []
                            continue
                        
                        # Check if this word can extend current phrase
                        can_extend = False
                        
                        # Rule 1: Noun sequences (N, Np, Nu, Nc, etc.)
                        if pos_tag.startswith('N'):
                            can_extend = True
                        
                        # Rule 2: Adjectives can extend if followed by nouns
                        elif pos_tag.startswith('A') and current_phrase:
                            can_extend = True
                        
                        # Rule 3: Proper nouns (Np) always start new phrase or extend
                        elif pos_tag == 'Np':
                            can_extend = True
                        
                        # Rule 4: Some specific words that can be part of phrases
                        elif word.lower() in ['cựu', 'nguyên', 'phó', 'thứ', 'đầu', 'tiên', 'chính', 'trưởng']:
                            can_extend = True
                        
                        # Rule 5: Numbers can be part of phrases in some contexts
                        elif pos_tag == 'M' and current_phrase:
                            can_extend = True
                        
                        if can_extend:
                            # Clean word: remove underscores for consistency
                            clean_word = word.replace('_', ' ')
                            current_phrase.append(clean_word)
                            current_pos_sequence.append(pos_tag)
                        else:
                            # Finish current phrase if it's meaningful
                            if current_phrase and len(current_phrase) >= 1:
                                phrase_text = " ".join(current_phrase)
                                if len(phrase_text) > 2 and phrase_text.lower() not in stop_words:
                                    phrases.append(phrase_text)
                            
                            # Start new phrase if this word can start one
                            if (pos_tag.startswith('N') or pos_tag == 'Np' or 
                                word.lower() in ['cựu', 'nguyên', 'phó', 'thứ', 'đầu', 'tiên', 'chính', 'trưởng']):
                                clean_word = word.replace('_', ' ')
                                current_phrase = [clean_word]
                                current_pos_sequence = [pos_tag]
                            else:
                                current_phrase = []
                                current_pos_sequence = []
                    
                    # Finish last phrase
                    if current_phrase and len(current_phrase) >= 1:
                        phrase_text = " ".join(current_phrase)
                        if len(phrase_text) > 2 and phrase_text.lower() not in stop_words:
                            phrases.append(phrase_text)
            
            else:
                # Fallback: simple approach with common Vietnamese phrase patterns
                import re
                
                # Simple patterns for common Vietnamese phrases
                phrase_patterns = [
                    r'(cựu|nguyên|phó)\s+\w+(\s+\w+)*',  # cựu tổng thống
                    r'[A-Z][a-z]+\s+[A-Z][a-z]+',        # Donald Trump  
                    r'\w+\s+(tổng thống|thủ tướng|chủ tịch|giám đốc)',  # noun + title
                    r'(đầu tiên|cuối cùng|duy nhất)\s+\w+',  # đầu tiên + noun
                ]
                
                for pattern in phrase_patterns:
                    matches = re.findall(pattern, claim_text, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            phrase = ' '.join(match)
                        else:
                            phrase = match
                        if phrase and len(phrase) > 2:
                            phrases.append(phrase)
        
        except Exception as e:
            print(f"⚠️ Error extracting phrases: {e}")
            return []
        
        # Remove duplicates and filter
        seen = set()
        unique_phrases = []
        for phrase in phrases:
            phrase_normalized = phrase.lower().strip()
            if (phrase_normalized not in seen and 
                len(phrase.strip()) > 2 and 
                not phrase.isdigit()):
                seen.add(phrase_normalized)
                unique_phrases.append(phrase.strip())
        
        print(f"✅ Extracted {len(unique_phrases)} phrase entities from claim: {unique_phrases}")
        return unique_phrases