#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MINT TextGraph - Beam Search Path Finding
Tìm đường đi từ claim đến sentence nodes bằng Beam Search
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import heapq
from datetime import datetime
import time
from difflib import SequenceMatcher
import networkx as nx


class Path:
    """Đại diện cho một đường đi trong đồ thị"""
    
    def __init__(self, nodes: List[str], edges: Optional[List[Tuple[str, str, str]]] = None, score: float = 0.0):
        self.nodes = nodes  # Danh sách node IDs
        self.edges = edges or []  # Danh sách (from_node, to_node, relation)
        self.score = score  # Điểm đánh giá path
        self.claim_words = set()  # Words trong claim để so sánh
        self.word_matches = set()  # ✅ THÊM: Set of matched words
        self.path_words = set()   # Từ trong path
        self.entities_visited = set()  # Entities đã đi qua
        
    def __lt__(self, other):
        """So sánh để sort paths theo score"""
        return self.score < other.score
        
    def add_node(self, node_id: str, edge_info: Optional[Tuple[str, str, str]] = None):
        """Thêm node vào path"""
        self.nodes.append(node_id)
        if edge_info:
            self.edges.append(edge_info)
            
    def copy(self):
        """Tạo bản copy của path"""
        new_path = Path(self.nodes.copy(), self.edges.copy(), self.score)
        new_path.claim_words = self.claim_words.copy()
        new_path.word_matches = self.word_matches.copy()
        new_path.path_words = self.path_words.copy()
        new_path.entities_visited = self.entities_visited.copy()
        return new_path
        
    def get_current_node(self):
        """Lấy node hiện tại (cuối path)"""
        return self.nodes[-1] if self.nodes else None
        
    def contains_node(self, node_id: str):
        """Kiểm tra path có chứa node này không"""
        return node_id in self.nodes
        
    def to_dict(self):
        """Convert path thành dictionary để export"""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'score': self.score,
            'length': len(self.nodes),
            'claim_words_matched': len(self.claim_words.intersection(self.path_words)),
            'total_claim_words': len(self.claim_words),
            'entities_visited': list(self.entities_visited),
            'path_summary': self._get_path_summary()
        }
        
    def _get_path_summary(self):
        """Tạo summary ngắn gọn của path"""
        node_types = []
        for node in self.nodes:
            if node.startswith('claim'):
                node_types.append('CLAIM')
            elif node.startswith('word'):
                node_types.append('WORD')
            elif node.startswith('sentence'):
                node_types.append('SENTENCE')
            elif node.startswith('entity'):
                node_types.append('ENTITY')
            else:
                node_types.append('UNKNOWN')
        return ' -> '.join(node_types)


class BeamSearchPathFinder:
    """Beam Search để tìm đường đi từ claim đến sentence nodes"""
    
    def __init__(self, text_graph, beam_width: int = 25, max_depth: int = 30, allow_skip_edge: bool = False):
        self.graph = text_graph
        self.beam_width = beam_width
        self.max_depth = max_depth
        # Cho phép "nhảy" qua một nút trung gian (2-hop) nếu cần mở rộng đa dạng
        self.allow_skip_edge = allow_skip_edge
        self.claim_words = set()  # Words trong claim
        
        # Scoring weights - ✅ CẢI THIỆN WEIGHTS
        self.word_match_weight = 5.0        # Tăng từ 3.0 lên 5.0
        self.semantic_match_weight = 3.0    # ✅ MỚI: Semantic similarity
        self.entity_bonus = 2.5             # Tăng từ 2.0 lên 2.5
        self.length_penalty = 0.05          # Giảm từ 0.1 xuống 0.05
        self.sentence_bonus = 4.0           
        self.fuzzy_match_weight = 2.0       # ✅ MỚI: Fuzzy string matching
        
        # Stats
        self.paths_explored = 0
        self.sentence_paths_found = 0
        
        # New flag
        self.early_stop_on_sentence = True
        
    def extract_claim_words(self):
        """Trích xuất tất cả từ trong claim để so sánh"""
        claim_words = set()
        
        if self.graph.claim_node:
            # Lấy tất cả word nodes connected đến claim
            for neighbor in self.graph.graph.neighbors(self.graph.claim_node):
                node_data = self.graph.graph.nodes[neighbor]
                if node_data.get('type') == 'word':
                    claim_words.add(node_data.get('text', '').lower())
                    
        self.claim_words = claim_words
        return claim_words
        
    def _calculate_semantic_similarity(self, claim_words, path_words):
        """
        ✅ MỚI: Tính semantic similarity giữa claim và path words
        Sử dụng Jaccard similarity và word overlap
        """
        if not claim_words or not path_words:
            return 0.0
            
        # Jaccard similarity
        intersection = claim_words.intersection(path_words)
        union = claim_words.union(path_words)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Word overlap ratio
        overlap_ratio = len(intersection) / len(claim_words) if claim_words else 0.0
        
        # Combine scores
        semantic_score = (jaccard * 0.4) + (overlap_ratio * 0.6)
        return semantic_score
        
    def _calculate_fuzzy_similarity(self, claim_text, sentence_text):
        """
        ✅ MỚI: Tính fuzzy string similarity
        """
        if not claim_text or not sentence_text:
            return 0.0
            
        # Normalize texts
        claim_normalized = claim_text.lower().strip()
        sentence_normalized = sentence_text.lower().strip()
        
        # Calculate similarity
        similarity = SequenceMatcher(None, claim_normalized, sentence_normalized).ratio()
        return similarity
        
    def score_path(self, path: Path) -> float:
        """✅ CẢI THIỆN: Tính điểm cho một path với nhiều metrics hơn"""
        
        if not path.nodes:
            return 0.0
            
        # Lấy claim text để so sánh
        claim_text = ""
        claim_words = set()
        
        for node in path.nodes:
            node_data = self.graph.graph.nodes[node]
            if node_data.get('type') == 'claim':
                claim_text = node_data.get('text', '')
                claim_words = set(claim_text.lower().split())
                break
                
        # Base score
        score = 0.0
        
        # 1. ✅ CẢI THIỆN: Enhanced Word matching score
        path_words = set()
        sentence_texts = []
        
        for node in path.nodes:
            node_data = self.graph.graph.nodes[node]
            node_text = node_data.get('text', '')
            node_type = node_data.get('type', '')
            
            if node_text:
                path_words.update(node_text.lower().split())
                
                # Collect sentence texts for fuzzy matching
                if node_type == 'sentence':
                    sentence_texts.append(node_text)
                
        if claim_words:
            word_matches = claim_words.intersection(path_words)
            word_match_ratio = len(word_matches) / len(claim_words)
            score += word_match_ratio * self.word_match_weight
            path.word_matches = word_matches
            
            # 2. ✅ MỚI: Semantic similarity
            semantic_score = self._calculate_semantic_similarity(claim_words, path_words)
            score += semantic_score * self.semantic_match_weight
            
        # 3. ✅ MỚI: Fuzzy matching với sentences
        if claim_text and sentence_texts:
            max_fuzzy_score = 0.0
            for sentence_text in sentence_texts:
                fuzzy_score = self._calculate_fuzzy_similarity(claim_text, sentence_text)
                max_fuzzy_score = max(max_fuzzy_score, fuzzy_score)
            score += max_fuzzy_score * self.fuzzy_match_weight
            
        # 4. ✅ ENHANCED: Dual-source entity scoring với weighted bonus
        entity_bonus_total = 0.0
        dual_source_entities = 0
        single_source_entities = 0
        
        for node in path.nodes:
            node_data = self.graph.graph.nodes[node]
            if node_data.get('type') == 'entity':
                # Get entity score (dual-source entities have score ≥ 2.0)
                entity_score = node_data.get('entity_score', 1.0)
                is_dual_source = node_data.get('is_dual_source', False)
                
                # All entities get equal treatment now
                entity_bonus_total += entity_score * self.entity_bonus
                if is_dual_source:
                    dual_source_entities += 1
                else:
                    single_source_entities += 1
                
        score += entity_bonus_total
        
        # Store entity path metadata for debugging
        path.dual_source_count = dual_source_entities
        path.single_source_count = single_source_entities
        # Keep entities_visited as set - don't overwrite with int
        
        # 5. ✅ CẢI THIỆN: Giảm length penalty
        score -= len(path.nodes) * self.length_penalty
        
        # 6. ✅ THÊM: Sentence relevance bonus
        sentence_count = sum(1 for node in path.nodes 
                           if self.graph.graph.nodes[node].get('type') == 'sentence')
        if sentence_count > 0:
            score += sentence_count * 1.5  # Bonus cho mỗi sentence trong path
            
        return score
        
    def beam_search(self, start_node: str = None) -> List[Path]:
        """
        Thực hiện Beam Search từ claim node đến sentence nodes
        
        Returns:
            List[Path]: Danh sách các paths tốt nhất tìm được
        """
        if start_node is None:
            start_node = self.graph.claim_node
            
        if not start_node:
            print("⚠️ Không tìm thấy claim node để bắt đầu beam search")
            return []
            
        # Extract claim words để scoring
        self.extract_claim_words()
        
        # Prepare graph data for faster lookup
        graph_data = dict(self.graph.graph.nodes(data=True))
        
        # Initialize beam với path từ claim node
        beam = [Path([start_node])]
        completed_paths = []  # Paths đã đến sentence nodes
        
        print(f"🎯 Starting Beam Search from {start_node}")
        print(f"📊 Beam width: {self.beam_width}, Max depth: {self.max_depth}")
        print(f"💭 Claim words: {self.claim_words}")
        
        for depth in range(self.max_depth):
            if not beam:
                break
                
            print(f"\n🔍 Depth {depth + 1}/{self.max_depth} - Current beam size: {len(beam)}")
            
            new_candidates = []
            
            # Expand mỗi path trong beam hiện tại
            for path in beam:
                current_node = path.get_current_node()
                
                # Lấy tất cả neighbors của current node
                neighbors = list(self.graph.graph.neighbors(current_node))
                
                for neighbor in neighbors:
                    # Tránh cycle - không quay lại node đã visit
                    if path.contains_node(neighbor):
                        continue
                        
                    # Tạo path mới
                    new_path = path.copy()
                    
                    # Lấy edge info
                    edge_data = self.graph.graph.get_edge_data(current_node, neighbor)
                    relation = edge_data.get('relation', 'unknown') if edge_data else 'unknown'
                    edge_info = (current_node, neighbor, relation)
                    
                    new_path.add_node(neighbor, edge_info)
                    
                    # Score path mới
                    new_path.score = self.score_path(new_path)
                    
                    # Kiểm tra nếu đạt sentence node
                    neighbor_data = graph_data.get(neighbor, {})
                    if neighbor_data.get('type') == 'sentence':
                        completed_paths.append(new_path)
                        print(f"  ✅ Found path to sentence: {neighbor} (score: {new_path.score:.3f})")
                    else:
                        new_candidates.append(new_path)
                        
            # Chọn top K candidates cho beam tiếp theo
            if new_candidates:
                # Sort by score descending và chọn top beam_width
                new_candidates.sort(key=lambda p: p.score, reverse=True)
                beam = new_candidates[:self.beam_width]
                
                # Debug info
                print(f"  📈 Top scores in beam: {[f'{p.score:.3f}' for p in beam[:5]]}")
            else:
                beam = []
                
        # Combine completed paths và sort theo score
        all_paths = completed_paths
        all_paths.sort(key=lambda p: p.score, reverse=True)
        
        print(f"\n🎉 Beam Search completed!")
        print(f"  Found {len(completed_paths)} paths to sentences")
        print(f"  Top path score: {all_paths[0].score:.3f}" if all_paths else "  No paths found")
        
        return all_paths
        
    def find_best_paths(self, max_paths: int = 20) -> List[Path]:
        """
        Tìm các path tốt nhất từ claim đến sentences
        
        Args:
            max_paths: Số lượng paths tối đa để trả về
            
        Returns:
            List[Path]: Danh sách paths được sắp xếp theo score
        """
        start_time = time.time()
        
        # Lấy claim nodes và sentence nodes  
        claim_nodes = [node for node, data in self.graph.graph.nodes(data=True) 
                      if data.get('type') == 'claim']
        sentence_nodes = [node for node, data in self.graph.graph.nodes(data=True)
                         if data.get('type') == 'sentence']
                         
        if not claim_nodes:
            print("⚠️  No claim nodes found!")
            return []
            
        if not sentence_nodes:
            print("⚠️  No sentence nodes found!")
            return []
            
        print(f"🎯 Found {len(claim_nodes)} claim nodes, {len(sentence_nodes)} sentence nodes")
        
        # Khởi tạo beam với paths từ mỗi claim node
        current_beam = []
        for claim_node in claim_nodes:
            initial_path = Path([claim_node], [], 0.0)
            current_beam.append(initial_path)
            
        completed_paths = []
        
        # Beam search main loop
        for depth in range(self.max_depth):
            if not current_beam:
                break
                
            next_beam = []
            
            for path in current_beam:
                current_node = path.nodes[-1]
                
                # Kiểm tra xem node hiện tại có phải sentence không
                current_node_data = self.graph.graph.nodes[current_node]
                if current_node_data.get('type') == 'sentence':
                    # Đã đến sentence node - có thể dừng ở đây
                    completed_paths.append(path)
                    self.sentence_paths_found += 1
                    continue  # Không expand thêm từ sentence node
                    
                # Expand path đến các neighbors
                for neighbor in self.graph.graph.neighbors(current_node):
                    # Tránh cycles
                    if neighbor in path.nodes:
                        continue
                        
                    # Tạo path mới
                    edge_data = self.graph.graph.get_edge_data(current_node, neighbor, {})
                    edge_label = edge_data.get('label', f"{current_node}->{neighbor}")
                    
                    new_path = Path(
                        path.nodes + [neighbor],
                        path.edges + [edge_label],
                        0.0
                    )
                    
                    # Tính điểm cho path mới
                    new_path.score = self.score_path(new_path)
                    next_beam.append(new_path)
                    self.paths_explored += 1
                    
            # Giữ lại top beam_width paths
            next_beam.sort(key=lambda p: p.score, reverse=True)
            current_beam = next_beam[:self.beam_width]
            
            if self.early_stop_on_sentence and completed_paths:
                break  # Dừng ngay khi tìm được sentence đầu tiên
            
        # Kết hợp completed paths và current beam
        all_paths = completed_paths + current_beam
        
        # Lọc chỉ lấy paths kết thúc tại sentence nodes
        sentence_paths = []
        for path in all_paths:
            if path.nodes:
                last_node = path.nodes[-1] 
                last_node_data = self.graph.graph.nodes[last_node]
                if last_node_data.get('type') == 'sentence':
                    sentence_paths.append(path)
                    
        # Sắp xếp và lấy top paths
        sentence_paths.sort(key=lambda p: p.score, reverse=True)
        
        end_time = time.time()
        print(f"⏱️  Beam search completed in {end_time - start_time:.2f}s")
        print(f"📊 Explored {self.paths_explored} paths, found {len(sentence_paths)} sentence paths")
        
        return sentence_paths[:max_paths]
        
    def export_paths_to_file(self, paths: List[Path], output_file: str = None) -> str:
        """
        Export paths ra file JSON để khảo sát
        
        Args:
            paths: Danh sách paths cần export
            output_file: Đường dẫn file output (nếu None sẽ tự generate)
            
        Returns:
            str: Đường dẫn file đã lưu
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use absolute path to ensure correct directory
            current_dir = os.getcwd()
            if current_dir.endswith('vncorenlp'):
                # If we're in vncorenlp directory, go back to parent
                current_dir = os.path.dirname(current_dir)
            output_file = os.path.join(current_dir, "output", f"beam_search_paths_{timestamp}.json")
            
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare data for export
        export_data = {
            'search_config': {
                'beam_width': self.beam_width,
                'max_depth': self.max_depth,
                'word_match_weight': self.word_match_weight,
                'entity_bonus': self.entity_bonus,
                'length_penalty': self.length_penalty,
                'sentence_bonus': self.sentence_bonus
            },
            'claim_words': list(self.claim_words),
            'total_paths_found': len(paths),
            'paths': []
        }
        
        # Prepare graph data for node details
        graph_data = dict(self.graph.graph.nodes(data=True))
        
        for i, path in enumerate(paths):
            path_data = path.to_dict()
            
            # Thêm thông tin chi tiết về nodes
            path_data['node_details'] = []
            for node_id in path.nodes:
                node_info = graph_data.get(node_id, {})
                path_data['node_details'].append({
                    'id': node_id,
                    'type': node_info.get('type', 'unknown'),
                    'text': node_info.get('text', ''),
                    'pos': node_info.get('pos', ''),
                    'lemma': node_info.get('lemma', '')
                })
                
            export_data['paths'].append(path_data)
            
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            
        print(f"💾 Exported {len(paths)} paths to: {output_file}")
        return output_file
        
    def export_paths_summary(self, paths: List[Path], output_file: str = None) -> str:
        """
        Export summary dễ đọc của paths
        
        Args:
            paths: Danh sách paths
            output_file: File output (nếu None sẽ tự generate)
            
        Returns:
            str: Đường dẫn file đã lưu
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use absolute path to ensure correct directory
            current_dir = os.getcwd()
            if current_dir.endswith('vncorenlp'):
                # If we're in vncorenlp directory, go back to parent
                current_dir = os.path.dirname(current_dir)
            output_file = os.path.join(current_dir, "output", f"beam_search_summary_{timestamp}.txt")
            
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare graph data
        graph_data = dict(self.graph.graph.nodes(data=True))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("🎯 BEAM SEARCH PATH ANALYSIS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Search Configuration:\n")
            f.write(f"  Beam Width: {self.beam_width}\n")
            f.write(f"  Max Depth: {self.max_depth}\n")
            f.write(f"  Claim Words: {', '.join(self.claim_words)}\n")
            f.write(f"  Total Paths Found: {len(paths)}\n\n")
            
            for i, path in enumerate(paths[:10]):  # Top 10 paths
                f.write(f"PATH #{i+1} (Score: {path.score:.3f})\n")
                f.write("-" * 40 + "\n")
                
                f.write(f"Length: {len(path.nodes)} nodes\n")
                f.write(f"Word Matches: {len(path.word_matches) if hasattr(path, 'word_matches') else 'None'}\n")
                f.write(f"Entities Visited: {', '.join(path.entities_visited) if path.entities_visited else 'None'}\n")
                f.write(f"Path Type: {path._get_path_summary()}\n\n")
                
                f.write("Detailed Path:\n")
                for j, node_id in enumerate(path.nodes):
                    node_info = graph_data.get(node_id, {})
                    node_type = node_info.get('type', 'unknown').upper()
                    node_text = node_info.get('text', '')[:50]  # Truncate long text
                    
                    prefix = "  START: " if j == 0 else f"  {j:2d}: "
                    f.write(f"{prefix}[{node_type}] {node_text}\n")
                    
                    if j < len(path.edges):
                        edge_info = path.edges[j]
                        f.write(f"       └─ ({edge_info[2]}) ─>\n")
                        
                f.write("\n" + "="*60 + "\n\n")
                
        print(f"📄 Exported paths summary to: {output_file}")
        return output_file

    def multi_level_beam_search(
        self,
        max_levels: int = 3,
        beam_width_per_level: int = 3,
        min_new_sentences: int = 2,   # ❶ bảo đảm mỗi level có ≥ 2 câu mới
        sbert_model=None,
        claim_text: str = "",
        entities=None,
        filter_top_k: int = 5,
        use_phobert: bool = False  # Use SBERT by default, PhoBERT if True
    ) -> Dict[int, List[Path]]:
        """
        Multi-level beam search: từ claim → sentences → sentences liên quan → ...
        
        Args:
            max_levels: Số levels tối đa (k)
            beam_width_per_level: Số sentences giữ lại mỗi level
            
        Returns:
            Dict[level, List[Path]]: Sentences theo từng level
        """
        results = {}
        all_found_sentences = set()  # Track sentences đã tìm để tránh trùng
        
        print(f"🎯 Starting Multi-Level Beam Search (max_levels={max_levels}, beam_width={beam_width_per_level})")
        
        # Level 0: Beam search từ claim
        print(f"\n📍 LEVEL 0: Claim → Sentences")
        level_0_paths = self.find_best_paths(max_paths=beam_width_per_level)
        level_0_sentences = self._extract_sentence_nodes_from_paths(level_0_paths)
        
        results[0] = level_0_paths
        all_found_sentences.update(level_0_sentences)
        
        print(f"   Found {len(level_0_sentences)} sentences at level 0")
        
        # Levels 1 to k: Beam search từ sentences của level trước
        current_sentence_nodes = level_0_sentences
        
        for level in range(1, max_levels + 1):
            if not current_sentence_nodes:
                print(f"   No sentences to expand from level {level-1}")
                break
                
            print(f"\n📍 LEVEL {level}: Sentences → New Sentences")
            level_paths = []
            new_sentence_nodes = set()
            
            # Beam search từ mỗi sentence của level trước
            for sentence_node in current_sentence_nodes:
                print(f"   Expanding from sentence: {sentence_node}")
                
                # Beam search từ sentence này
                sentence_paths = self._beam_search_from_sentence(
                    sentence_node, 
                    max_paths=beam_width_per_level,
                    exclude_sentences=all_found_sentences
                )
                
                # Lấy sentences mới
                new_sentences = self._extract_sentence_nodes_from_paths(sentence_paths)
                new_sentences = [s for s in new_sentences if s not in all_found_sentences]
                
                level_paths.extend(sentence_paths)
                new_sentence_nodes.update(new_sentences)
                
                print(f"     → Found {len(new_sentences)} new sentences")
            
            # Giữ lại top beam_width_per_level sentences tốt nhất cho level này
            if level_paths:
                level_paths.sort(key=lambda p: p.score, reverse=True)
                level_paths = level_paths[:beam_width_per_level]

                # ❷ Lấy câu mới, loại trùng
                final_new_sentences = self._extract_sentence_nodes_from_paths(level_paths)
                unique_new = [s for s in final_new_sentences if s not in all_found_sentences]

                # 🔍 Apply SBERT/PhoBERT filtering at each level
                if sbert_model and claim_text and unique_new:
                    try:
                        # Get sentence texts from nodes
                        sentence_texts = []
                        for node in unique_new:
                            node_text = self.graph.graph.nodes[node].get("text", "")
                            if node_text:
                                sentence_texts.append(node_text)
                        
                        if sentence_texts:
                            # Calculate similarities using SBERT or PhoBERT
                            if use_phobert and hasattr(self.graph, 'get_sentence_similarity'):
                                # Use PhoBERT via TextGraph method
                                print(f"   🔍 Using PhoBERT for level {level} filtering...")
                                similarities = []
                                for sent_text in sentence_texts:
                                    try:
                                        sim = self.graph.get_sentence_similarity(sent_text, claim_text)
                                        similarities.append(sim)
                                    except Exception as e:
                                        print(f"   ⚠️ PhoBERT similarity error: {e}")
                                        similarities.append(0.0)  # Fallback score
                            else:
                                # Use SBERT
                                print(f"   🔍 Using SBERT for level {level} filtering...")
                                from sklearn.metrics.pairwise import cosine_similarity
                                claim_embedding = sbert_model.encode([claim_text])
                                sentence_embeddings = sbert_model.encode(sentence_texts)
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
                            model_name = "PhoBERT" if use_phobert else "SBERT"
                            print(f"   🔍 {model_name} level filtering: {len(unique_new)} sentences retained")
                    except Exception as e:
                        print(f"   ⚠️ Level filtering error: {e}")
                        # Continue with all sentences if filtering fails
                else:
                    print(f"   📦 Collected {len(unique_new)} raw sentences at level {level} (no level filtering)")

                # ❸ Nếu chưa đủ, lấy thêm câu (không trùng) từ danh sách level_paths (đã xếp hạng)
                if len(unique_new) < min_new_sentences:
                    for path in level_paths:
                        for node in path.nodes[::-1]:  # duyệt từ cuối path
                            node_data = self.graph.graph.nodes[node]
                            if node_data.get('type') == 'sentence' and node not in all_found_sentences:
                                unique_new.append(node)
                                if len(unique_new) >= min_new_sentences:
                                    break
                        if len(unique_new) >= min_new_sentences:
                            break

                # ❹ Cập nhật kết quả / tracking
                results[level] = level_paths
                all_found_sentences.update(unique_new)
                current_sentence_nodes = unique_new
                
                print(f"   Level {level} final: {len(unique_new)} sentences")
            else:
                print(f"   Level {level}: No new sentences found")
                break
        
        print(f"\n🎉 Multi-Level Search completed! Total levels: {len(results)}")
        return results

    def _extract_sentence_nodes_from_paths(self, paths: List[Path]) -> List[str]:
        """Extract unique sentence node IDs từ paths"""
        sentence_nodes = set()
        for path in paths:
            for node in path.nodes:
                node_data = self.graph.graph.nodes.get(node, {})
                if node_data.get('type') == 'sentence':
                    sentence_nodes.add(node)
        return list(sentence_nodes)

    def _beam_search_from_sentence(self, start_sentence: str, max_paths: int = 3, exclude_sentences: Set[str] = None) -> List[Path]:
        """
        Beam search từ một sentence node để tìm sentences liên quan
        
        Args:
            start_sentence: Sentence node để bắt đầu
            max_paths: Số paths tối đa
            exclude_sentences: Sentences cần loại trừ (đã tìm trước đó)
        """
        if exclude_sentences is None:
            exclude_sentences = set()
        
        # Initialize beam từ sentence node
        beam = [Path([start_sentence])]
        completed_paths = []
        
        # Reduced depth for sentence-to-sentence search
        max_depth = min(self.max_depth // 2, 15)  # Shorter paths for efficiency
        
        for depth in range(max_depth):
            if not beam:
                break
                
            new_candidates = []
            
            for path in beam:
                current_node = path.get_current_node()
                
                # Expand to neighbors
                for neighbor in self.graph.graph.neighbors(current_node):
                    # Avoid cycles
                    if path.contains_node(neighbor):
                        continue
                    
                    # Tạo path mới
                    new_path = path.copy()
                    edge_data = self.graph.graph.get_edge_data(current_node, neighbor)
                    relation = edge_data.get('relation', 'unknown') if edge_data else 'unknown'
                    edge_info = (current_node, neighbor, relation)
                    
                    new_path.add_node(neighbor, edge_info)
                    new_path.score = self.score_path(new_path)
                    
                    # Check if reached new sentence
                    neighbor_data = self.graph.graph.nodes.get(neighbor, {})
                    if (neighbor_data.get('type') == 'sentence' and 
                        neighbor != start_sentence and  # Not same as start
                        neighbor not in exclude_sentences):  # Not already found
                        completed_paths.append(new_path)
                    else:
                        new_candidates.append(new_path)
            
            # Keep top candidates
            if new_candidates:
                new_candidates.sort(key=lambda p: p.score, reverse=True)
                beam = new_candidates[:self.beam_width]
            else:
                beam = []
        
        # Return top sentence paths
        completed_paths.sort(key=lambda p: p.score, reverse=True)
        return completed_paths[:max_paths] 