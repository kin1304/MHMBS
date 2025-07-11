#!/usr/bin/env python3
"""
🚀 MULTI-HOP MULTI-BEAM SEARCH
==============================

Multi-hop beam search pipeline thực hiện nhiều lần quá trình:
📊 Input Data → 🌱 Beam Search → 🏷️ Entity Extraction → 🔍 Advanced Filtering → 📊 SBERT Reranking

Pipeline Logic:
- Hop 1: Chạy đầy đủ pipeline 
- Hop 2-N: Sử dụng kết quả từ hop trước làm start nodes, tận dụng lại VnCoreNLP và entities

Features:
- Configurable số hops (default: 3)
- Reuse VnCoreNLP processing và entity extraction
- Progressive knowledge accumulation
- Enhanced multi-hop evidence discovery

Author: AI Assistant & NguyenNha
Date: 2025-01-03
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

# Add vncorenlp to path
sys.path.append("/Users/nguyennha/Desktop/project_multi_hop/vncorenlp")
import py_vncorenlp

# Add mint package
from mint.text_graph import TextGraph
from mint.beam_search import Path as BeamSearchPath

# Import advanced filtering
from advanced_data_filtering import AdvancedDataFilter, integrate_advanced_filtering_with_existing_pipeline

# Try to import SBERT
SBERT_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
    print("✅ SBERT components loaded")
except ImportError as e:
    print(f"⚠️ SBERT not available: {e}")


class MultiHopMultiBeamProcessor:
    """
    🚀 Multi-Hop Multi-Beam Search Processor
    
    Thực hiện nhiều hop của beam search pipeline, mỗi hop tận dụng kết quả từ hop trước
    """
    
    def __init__(self, num_hops: int = 3, use_advanced_filtering=True, use_sbert=True, 
                 use_contradiction_detection=True, use_entity_filtering=True,
                 min_quality_score: float = 0.3,
                 min_relevance_score: float = 0.15, target_ratio: float = 0.5,
                 min_entity_score: float = 0.05,
                 stance_delta: float = 0.1,
                 require_subject_match: bool = False,
                 hop_decay_factor: float = 0.8):
        """
        Initialize Multi-Hop Processor
        
        Args:
            num_hops: Số lượng hops (default: 3)
            hop_decay_factor: Factor giảm beam width sau mỗi hop (default: 0.8)
        """
        self.num_hops = max(1, num_hops)
        self.hop_decay_factor = max(0.1, min(1.0, hop_decay_factor))
        
        self.use_advanced_filtering = use_advanced_filtering
        self.use_sbert = use_sbert and SBERT_AVAILABLE
        self.use_contradiction_detection = use_contradiction_detection
        self.use_entity_filtering = use_entity_filtering
        
        # Store quality & relevance thresholds
        self.min_quality_score = max(0.0, min_quality_score)
        self.min_entity_score = max(0.0, min_entity_score)
        self.stance_delta = max(0.0, stance_delta)
        self.require_subject_match = require_subject_match
        
        if not (use_sbert and SBERT_AVAILABLE):
            self.min_relevance_score = max(0.05, min_relevance_score)
        else:
            self.min_relevance_score = min_relevance_score

        self.target_ratio = max(0.1, min(target_ratio, 1.0))
        
        # Initialize advanced filter
        if self.use_advanced_filtering:
            self.advanced_filter = AdvancedDataFilter(
                use_sbert=self.use_sbert,
                use_contradiction_detection=self.use_contradiction_detection
            )
        else:
            self.advanced_filter = None
        
        # Initialize SBERT if available
        if self.use_sbert:
            try:
                self.sbert_model = SentenceTransformer("keepitreal/vietnamese-sbert")
                print("✅ SBERT model loaded")
            except Exception as e:
                print(f"⚠️ SBERT failed, using fallback: {e}")
                try:
                    self.sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                    print("✅ Fallback SBERT model loaded")
                except Exception as e2:
                    print(f"❌ All SBERT models failed: {e2}")
                    self.use_sbert = False
                    self.sbert_model = None
        else:
            self.sbert_model = None
        
        # Global graph for reuse across samples
        self.global_graph = None
        
        print(f"🔧 Multi-Hop Multi-Beam Processor initialized:")
        print(f"   - Number of Hops: {self.num_hops}")
        print(f"   - Hop Decay Factor: {self.hop_decay_factor}")
        print(f"   - Advanced Filtering: {'✅' if self.use_advanced_filtering else '❌'}")
        print(f"   - SBERT Semantic Filtering: {'✅' if self.use_sbert else '❌'}")
        print(f"   - Contradiction Detection: {'✅' if self.use_contradiction_detection else '❌'}")
        print(f"   - Entity-Based Filtering: {'✅' if self.use_entity_filtering else '❌'}")

    def process_multi_hop_search(self, text_graph: TextGraph, claim_text: str,
                                max_levels: int = 3, beam_width_per_level: int = 6,
                                max_depth: int = 30, max_final_sentences: int = 30,
                                initial_entities: List[str] | None = None) -> Dict:
        """
        🚀 Process Multi-Hop Multi-Beam Search
        
        Returns:
            Dict containing:
            - hop_results: List of results from each hop
            - aggregated_sentences: Combined results from all hops
            - final_sentences: Final filtered and ranked sentences
            - comprehensive_statistics: Detailed stats across all hops
        """
        print(f"🔍 Starting Multi-Hop Multi-Beam Search với {self.num_hops} hops...")
        print(f"📄 Claim: {claim_text[:100]}...")
        
        # Storage for all hop results
        hop_results = []
        all_accumulated_sentences = []
        # Reuse entities đã extract ở bước chuẩn bị (HOP 1)
        reused_entities: List[str] = list(initial_entities) if initial_entities else []
        
        # Context sentences for reuse (extracted once)
        context_sentences: List[str] = []
        
        print(f"\n🔍 DEBUG: Starting multi-hop process with num_hops={self.num_hops}")
        print(f"🔍 DEBUG: Range will be: {list(range(1, self.num_hops + 1))}")
        
        for hop_num in range(1, self.num_hops + 1):
            print(f"\n🚀 === HOP {hop_num}/{self.num_hops} ===")
            print(f"🔍 DEBUG: Starting HOP {hop_num}")
            
            # Use fixed parameters (disable hop decay for testing)
            current_beam_width = beam_width_per_level
            current_max_depth = max_depth
            
            print(f"⚙️ Hop {hop_num} Parameters: beam_width={current_beam_width}, max_depth={current_max_depth}")
            
            if hop_num == 1:
                # HOP 1: Traditional full pipeline
                hop_result = self._process_first_hop(
                    text_graph, claim_text,
                    max_levels, current_beam_width, current_max_depth,
                    max_final_sentences, pre_entities=reused_entities
                )
                
                # Store context sentences and entities for reuse
                context_sentences_raw = hop_result.get('context_sentences', [])
                context_sentences: List[str] = []
                if isinstance(context_sentences_raw, list):
                    context_sentences = [str(s) for s in context_sentences_raw if s]
                
                reused_entities_raw = hop_result.get('entities', [])
                reused_entities: List[str] = []
                if isinstance(reused_entities_raw, list):
                    reused_entities = [str(e) for e in reused_entities_raw if e]
                
            else:
                # HOP 2+: Use previous hop results as start nodes
                previous_sentences = hop_results[-1]['filtered_sentences']
                hop_result = self._process_subsequent_hop(
                    text_graph, claim_text, previous_sentences, 
                    context_sentences, reused_entities,
                    hop_num, max_levels, current_beam_width, 
                    current_max_depth, max_final_sentences,
                    all_accumulated_sentences
                )
            
            hop_results.append(hop_result)
            all_accumulated_sentences.extend(hop_result['filtered_sentences'])
            
            # 🔄 Entities remain unchanged between hops (reuse existing entities)
            print(f"✅ Hop {hop_num} completed: {len(hop_result['filtered_sentences'])} sentences, reusing {len(reused_entities)} entities")
            print(f"🔍 DEBUG: Completed HOP {hop_num}, continuing to next hop...")
        
        print(f"🔍 DEBUG: Finished all {self.num_hops} hops, proceeding to final aggregation")
        
        # FINAL AGGREGATION AND RANKING
        print(f"\n🔄 Aggregating results from {self.num_hops} hops...")
        final_result = self._aggregate_and_rank_final_results(
            hop_results, all_accumulated_sentences, claim_text, 
            text_graph, max_final_sentences
        )
        
        return final_result

    def _process_first_hop(self, text_graph: TextGraph, claim_text: str,
                          max_levels: int, beam_width: int, max_depth: int,
                          max_final_sentences: int, pre_entities: List[str] | None = None) -> Dict:
        """
        🌱 Process First Hop - Có thể dùng Direct Entity Connection hoặc Traditional Beam Search
        """
        
        # Stage 1A: Direct Entity Connection (Fast & High Precision)
        high_confidence_sentences = []  # Fallback to simple approach
        
        # Convert direct connection results thành format tương thích
        direct_sentences = []
        for i, sent_data in enumerate(high_confidence_sentences):
            confidence_scores = {
                'very_high': 0.95,
                'high': 0.85, 
                'medium': 0.65,
                'low': 0.45
            }
            
            sentence_info = {
                'sentence': sent_data['text'],
                'score': confidence_scores.get(sent_data['confidence'], 0.5),
                'level': 0,
                'path_length': 1,
                'source': 'direct_entity_connection',
                'confidence': sent_data['confidence'],
                'connected_to_entities': sent_data['connected_to_entities'],
                'similarity_score': sent_data['similarity_score']
            }
            direct_sentences.append(sentence_info)
        
        # Stage 1B: Traditional Beam Search (Comprehensive Coverage)
        multi_results = text_graph.multi_level_beam_search_paths(
            max_levels=max_levels,
            beam_width_per_level=beam_width,
            max_depth=max_depth
        )
        
        # ✅ STAGE 1C: Extract và merge beam search results
        # print("   🔗 Stage 1C: Merging Direct Connection + Beam Search results...")
        beam_sentences = []
        sentence_score_map = {}  # sentence_text -> {sentence_data, scores[], paths_count}
        
        for level, paths in multi_results.items():
            level_sentences = self._extract_sentences_from_paths(paths, text_graph, level)
            
            # Validate that level_sentences is a list of dictionaries
            if not isinstance(level_sentences, list):
                print(f"⚠️ Warning: level_sentences is not a list: {type(level_sentences)}")
                continue
                
            # Aggregate scores for duplicate sentences
            for sent in level_sentences:
                # Validate that sent is a dictionary
                if not isinstance(sent, dict):
                    print(f"⚠️ Warning: sent in level_sentences is not a dict: {type(sent)}, value: {sent}")
                    continue
                    
                sent_text = sent.get('sentence', '').strip()
                if sent_text and len(sent_text) > 10:
                    if sent_text in sentence_score_map:
                        # Aggregate: add score và tăng paths_count
                        sentence_score_map[sent_text]['scores'].append(sent.get('score', 0))
                        sentence_score_map[sent_text]['paths_count'] += 1
                        # Update aggregated score (average với paths_count weighting)
                        scores = sentence_score_map[sent_text]['scores']
                        paths_count = sentence_score_map[sent_text]['paths_count']
                        # Weighted score: average + controlled bonus for multiple paths
                        avg_score = sum(scores) / len(scores)
                        # Logarithmic bonus để tránh over-reward duplicates
                        import math
                        paths_bonus = 0.05 * math.log(paths_count)  # Logarithmic scaling
                        paths_bonus = min(paths_bonus, 0.2)  # Max 20% bonus
                        sentence_score_map[sent_text]['sentence_data']['score'] = avg_score + paths_bonus
                        sentence_score_map[sent_text]['sentence_data']['multi_path_info'] = {
                            'paths_count': paths_count,
                            'score_range': [min(scores), max(scores)],
                            'average_score': avg_score,
                            'paths_bonus': paths_bonus
                        }
                    else:
                        # First time seeing this sentence
                        # Validate sent is a dictionary
                        if not isinstance(sent, dict):
                            print(f"⚠️ Warning: sent is not a dict: {type(sent)}, value: {sent}")
                            continue
                            
                        # Apply keyword boosting for important terms from claim
                        original_score = sent.get('score', 0)
                        boosted_score = self._apply_keyword_boosting(sent_text, claim_text, original_score)
                        sent['score'] = boosted_score
                        sent['source'] = 'beam_search'  # Mark source
                        
                        sentence_score_map[sent_text] = {
                            'sentence_data': sent.copy(),
                            'scores': [boosted_score],
                            'paths_count': 1
                        }
        
        # Convert beam search map back to list
        beam_sentences = [data['sentence_data'] for data in sentence_score_map.values()]
        print(f"   ✅ Beam Search: Found {len(beam_sentences)} sentences from {len(multi_results)} levels")
        
        # ✅ STAGE 1D: Smart Merge - Prioritize Direct Connection, Add Unique Beam Search
        print("   🎯 Stage 1D: Smart Merging with Deduplication...")
        
        # Start with direct connection sentences (high priority)
        # Validate direct_sentences is a list
        if not isinstance(direct_sentences, list):
            print(f"⚠️ Warning: direct_sentences is not a list: {type(direct_sentences)}, value: {direct_sentences}")
            direct_sentences = []
            
        all_sentences = direct_sentences.copy()
        direct_texts = set()
        for sent in direct_sentences:
            # Normalize text for comparison
            normalized_text = sent['sentence'].replace('_', ' ').strip().lower()
            direct_texts.add(normalized_text)
        
        # Add beam search sentences - MORE INCLUSIVE approach
        added_from_beam = 0
        for beam_sent in beam_sentences:
            beam_normalized = beam_sent['sentence'].replace('_', ' ').strip().lower()
            
            # Check for EXACT duplicates only (not similarity)
            is_exact_duplicate = False
            for direct_sent in direct_sentences:
                direct_normalized = direct_sent['sentence'].replace('_', ' ').strip().lower()
                if direct_normalized == beam_normalized:
                    # EXACT match found - boost the direct connection version
                    if direct_sent.get('score', 0) < 0.9:
                        direct_sent['score'] = min(0.95, direct_sent.get('score', 0) + 0.1)
                    direct_sent['source'] = 'hybrid_both_methods'
                    direct_sent['beam_search_confirmed'] = True
                    is_exact_duplicate = True
                    break
            
            # If not exact duplicate, add beam search sentence  
            if not is_exact_duplicate:
                beam_sent['source'] = 'beam_search_unique'
                all_sentences.append(beam_sent)
                added_from_beam += 1
        
        print(f"   ✅ Hybrid Result: {len(direct_sentences)} direct + {added_from_beam} unique beam = {len(all_sentences)} total")
        
        # Sort by score descending để ưu tiên sentences tốt nhất
        all_sentences.sort(key=lambda x: x.get('score', 0), reverse=True)
        

        
        # 🚨 FALLBACK: Nếu beam search không tìm được sentences, lấy từ context
        if len(all_sentences) == 0:
            print("⚠️ Hop 1: No sentences from beam search - using context sentences as fallback")
            try:
                context_sentences_fallback = []
                for _, node_data in text_graph.graph.nodes(data=True):
                    if node_data.get('type') == 'sentence' and 'sentence' in node_data:
                        sentence_text = node_data['sentence']
                        sentence_info = {
                            'sentence': sentence_text,
                            'score': 0.5,  # Default score
                            'level': 0,
                            'path_length': 1,
                            'source': 'context_fallback'
                        }
                        context_sentences_fallback.append(sentence_info)
                
                all_sentences = context_sentences_fallback[:10]  # Lấy tối đa 10 sentences
                print(f"✅ Hop 1: Using {len(all_sentences)} context sentences as fallback")
            except Exception as e:
                print(f"❌ Hop 1: Context fallback failed: {e}")
                all_sentences = []
        
        # STAGE 2: ENTITY SETUP (Reuse if provided, otherwise extract once)
        if pre_entities:
            entities = pre_entities  # Reuse pre-extracted entities
            print(f"🏷️ Hop 1: Reusing {len(entities)} pre-extracted entities (no new extraction)")
            entity_nodes = []  # Skip adding – already in graph
        else:
            entities = []
        if not pre_entities and self.use_entity_filtering:
            try:
                print(f"🏷️ Hop 1: Starting FULL entity extraction...")
                
                # Get context text from graph
                context_text = self._get_context_text(text_graph)
                
                # 1. Extract phrase entities từ claim
                claim_phrase_entities = text_graph.extract_phrases_from_claim(
                    claim_text=claim_text,
                    claim_sentences=text_graph.claim_sentences if hasattr(text_graph, 'claim_sentences') else None
                )
                
                # 2. Extract entities with OpenAI
                openai_entities = []
                if context_text and len(context_text) > 50:
                    openai_entities = text_graph.extract_entities_with_openai(context_text + "\n" + claim_text)
                
                # 3. Merge and deduplicate entities
                all_entities = claim_phrase_entities + openai_entities
                seen = set()
                entities = []
                for entity in all_entities:
                    entity_normalized = entity.lower().strip()
                    if entity_normalized not in seen and len(entity.strip()) > 2:
                        seen.add(entity_normalized)
                        entities.append(entity.strip())
                
                # 4. Add entities to graph
                if entities:
                    text_graph.add_entities_to_graph(
                        entities=entities,
                        context_sentences=text_graph.context_sentences if hasattr(text_graph, 'context_sentences') else {}
                    )
                    
                print(f"✅ Hop 1: Extracted {len(entities)} entities and added to graph")
                
            except Exception as e:
                print(f"⚠️ Hop 1: Entity extraction failed: {e}")
                entities = []
        
        # STAGE 3: Hop 1 - Advanced filtering với điều kiện DỄ HƠN
        if self.use_advanced_filtering and self.advanced_filter:
            print("🔍 Hop 1: Advanced filtering with relaxed conditions...")
            # Apply advanced filtering với thresholds thấp hơn cho hop 1
            original_min_quality = self.min_quality_score
            original_min_relevance = self.min_relevance_score
            
            # Temporarily lower thresholds for hop 1
            self.min_quality_score = max(0.1, self.min_quality_score - 0.2)  # Giảm 0.2 points
            self.min_relevance_score = max(0.05, self.min_relevance_score - 0.15)  # Giảm 0.15 points
            
            print(f"🔧 Hop 1 relaxed thresholds: quality≥{self.min_quality_score:.2f}, relevance≥{self.min_relevance_score:.2f}")
            
            filtered_sentences = self._apply_advanced_filtering(
                all_sentences, claim_text, entities, len(all_sentences), hop_num=1
            )
            
            # Restore original thresholds
            self.min_quality_score = original_min_quality
            self.min_relevance_score = original_min_relevance
            
            print(f"✅ Hop 1: Kept {len(filtered_sentences)}/{len(all_sentences)} sentences (relaxed advanced filtering)")
        else:
            filtered_sentences = all_sentences
            print(f"✅ Hop 1: Kept all {len(filtered_sentences)} sentences (filtering disabled)")
        
        # STAGE 4: Final PhoBERT Reranking
        if self.use_sbert and self.sbert_model:
            # Sử dụng SBERT cho final reranking
            filtered_sentences = self._apply_final_sbert_reranking(
                filtered_sentences, claim_text, 
                use_phobert=False,  # ✅ Dùng SBERT
                text_graph=text_graph
            )
        
        # Get context sentences for reuse
        try:
            context_sentences = []
            for _, node_data in text_graph.graph.nodes(data=True):
                if node_data.get('type') == 'sentence' and 'sentence' in node_data:
                    context_sentences.append(node_data['sentence'])
        except Exception:
            context_sentences = []
        
        return {
            'hop_number': 1,
            'beam_search_results': multi_results,
            'raw_sentences': all_sentences,
            'entities': entities,
            'filtered_sentences': filtered_sentences,
            'context_sentences': context_sentences,
            'statistics': {
                'raw_count': len(all_sentences),
                'filtered_count': len(filtered_sentences),
                'entities_count': len(entities)
            }
        }

    def _process_subsequent_hop(self, text_graph: TextGraph, claim_text: str,
                               previous_sentences: List[Dict], 
                               context_sentences: List[str],
                               reused_entities: List[str],
                               hop_num: int, max_levels: int, beam_width: int, 
                               max_depth: int, max_final_sentences: int,
                               all_accumulated_sentences: List[Dict]) -> Dict:
        """
        🔄 HOP 2-N: Individual Loop Pipeline
        For EACH sentence from previous hop: run complete individual pipeline
        """
        print(f"🔄 HOP {hop_num}: Individual Loop Pipeline")
        print(f"📋 Processing {len(previous_sentences)} sentences from previous hop...")
        
        try:
            # Track all sentences already found để tránh trùng lặp
            existing_sentence_texts = set()
            for sent_data in all_accumulated_sentences:
                sentence_text = sent_data['sentence'].replace('_', ' ').strip().lower()
                existing_sentence_texts.add(sentence_text)
            
            # 🏷️ Stage 2: Reuse Entities (không extract lại)
            print(f"🏷️ Stage 2: Reusing {len(reused_entities)} entities from HOP 1...")
            updated_entities = reused_entities  # Keep entities unchanged
                
            # 📋 For EACH sentence from previous hop: Individual Pipeline
            all_new_sentences = []
            individual_results = []
            
            for i, sent_data in enumerate(previous_sentences, 1):
                sentence_text = sent_data['sentence'].replace('_', ' ').strip()
                if not sentence_text or len(sentence_text) <= 10:
                    continue
                    
                print(f"\n📋 Sentence {i}/{len(previous_sentences)}: {sentence_text[:80]}...")
                
                # 🌱 Stage 1: Beam Search (từ sentence này)
                print(f"   🌱 Stage 1: Beam Search from this sentence...")
                
                # Tìm sentence node ID từ sentence text
                start_node_id = None
                for node_id, node_data in text_graph.graph.nodes(data=True):
                    if (node_data.get('type') == 'sentence' and 
                        node_data.get('text', '').replace('_', ' ').strip() == sentence_text.replace('_', ' ').strip()):
                        start_node_id = node_id
                        break
                
                if not start_node_id:
                    print(f"      ❌ Could not find node ID for sentence: {sentence_text[:50]}...")
                    continue
                    
                individual_multi_results = text_graph.multi_level_beam_search_paths(
                    max_levels=max_levels,
                    beam_width_per_level=beam_width,
                    max_depth=max_depth,
                    claim_text=claim_text,
                    entities=updated_entities  # Use updated entities with new ones from evidence
                )
                
                # Pass use_phobert option to beam search if available
                if hasattr(self, 'use_phobert_level_filtering'):
                    text_graph.use_phobert_level_filtering = self.use_phobert_level_filtering
                
                # Extract sentences từ beam search results
                individual_sentences = []
                local_seen = set()
                for level, paths in individual_multi_results.items():
                    level_sentences = self._extract_sentences_from_paths(paths, text_graph, level)
                    # Deduplicate within this individual search
                    for sent in level_sentences:
                        # Validate sent is a dictionary
                        if not isinstance(sent, dict):
                            print(f"      ⚠️ Warning: sent in individual level_sentences is not a dict: {type(sent)}, value: {sent}")
                            continue
                            
                        sent_text = sent.get('sentence', '').strip()
                        if sent_text and sent_text not in local_seen:
                            local_seen.add(sent_text)
                            individual_sentences.append(sent)
                
                print(f"      ✅ Found {len(individual_sentences)} sentences from beam search")
                
                # 🏷️ Stage 2: Reuse Entities (không extract lại)
                print(f"   🏷️ Stage 2: Reusing {len(updated_entities)} entities from HOP 1")
                

                
                # Filter out duplicates với sentences đã có
                new_sentences_from_this_node = []
                for sent_info in individual_sentences:
                    sent_text_normalized = sent_info['sentence'].replace('_', ' ').strip().lower()
                    if (sent_text_normalized not in existing_sentence_texts and 
                        len(sent_text_normalized) > 10):
                        new_sentences_from_this_node.append(sent_info)
                        existing_sentence_texts.add(sent_text_normalized)  # Add to tracking set
                

                
                # STAGE 2: Collect raw sentences without filtering
                if new_sentences_from_this_node:
                    # Add hop number to each sentence without individual filtering
                    for sent_info in new_sentences_from_this_node:
                        sent_info['hop_number'] = hop_num
                        sent_info['start_node_index'] = i
                        sent_info['start_node_text'] = sentence_text[:50] + "..."
                    
                    all_new_sentences.extend(new_sentences_from_this_node)
                    print(f"   📦 Collected: {len(new_sentences_from_this_node)} raw sentences from this node")
                
                # Store individual result for debugging
                individual_results.append({
                    'start_node_index': i,
                    'start_node_text': sentence_text,
                    'raw_count': len(individual_sentences),
                    'new_unique_count': len(new_sentences_from_this_node),
                    'final_collected_count': len(new_sentences_from_this_node)
                })
            
            # STAGE 4: Apply Advanced Filtering AFTER beam search complete
            if all_new_sentences and self.use_advanced_filtering and self.advanced_filter:
                print(f"🔍 Hop {hop_num}: Applying advanced filtering to {len(all_new_sentences)} raw sentences...")
                filtered_sentences = self._apply_advanced_filtering(
                    all_new_sentences, claim_text, updated_entities,  # Use updated entities with new ones
                    max_final_sentences, hop_num=hop_num
                )
                print(f"✅ Hop {hop_num}: Advanced filter kept {len(filtered_sentences)}/{len(all_new_sentences)} sentences")
            else:
                filtered_sentences = all_new_sentences
                print(f"✅ Hop {hop_num}: Kept all {len(filtered_sentences)} sentences (filtering disabled)")
            
            # STAGE 5: Apply SBERT reranking AFTER filtering
            if filtered_sentences and self.use_sbert and self.sbert_model:
                print(f"🔄 Hop {hop_num}: Applying SBERT reranking to {len(filtered_sentences)} sentences...")
                filtered_sentences = self._apply_final_sbert_reranking(
                    filtered_sentences, claim_text,
                    use_phobert=False,  # ✅ Dùng SBERT
                    text_graph=text_graph
                )
                print(f"✅ Hop {hop_num}: SBERT reranking complete")

            # STAGE 6: Final sorting và limiting - Hop 2+ chỉ lấy 3-4 sentences
            if filtered_sentences:
                # Sort by score and limit
                filtered_sentences.sort(key=lambda x: x.get('score', 0), reverse=True)
                # Hop 2+ chỉ lấy tối đa 4 sentences
                hop_limit = 4 if hop_num > 1 else max_final_sentences
                final_limited_sentences = filtered_sentences[:hop_limit]
                print(f"✅ Hop {hop_num}: Limited to {len(final_limited_sentences)} sentences (max {hop_limit} for hop {hop_num})")
            else:
                final_limited_sentences = []
            
            return {
                'hop_number': hop_num,
                'start_nodes_count': len(previous_sentences),
                'individual_results': individual_results,
                'raw_sentences': all_new_sentences,  # All new sentences before limiting
                'entities': updated_entities,  # Updated entities including new ones from evidence
                'filtered_sentences': final_limited_sentences,  # Final limited sentences
                'statistics': {
                    'start_nodes_processed': len(previous_sentences),
                    'raw_count': len(all_new_sentences),
                    'filtered_count': len(final_limited_sentences),
                    'entities_count': len(updated_entities),
                    'duplicate_avoidance_count': len(existing_sentence_texts),
                    'entities_reused': len(reused_entities)  # Track reused entities instead
                }
            }
        except Exception as e:
            print(f"❌ Error in _process_subsequent_hop (Hop {hop_num}): {e}")
            import traceback
            traceback.print_exc()
            # Return empty result
            return {
                'hop_number': hop_num,
                'start_nodes_count': len(previous_sentences) if 'previous_sentences' in locals() else 0,
                'individual_results': [],
                'raw_sentences': [],
                'entities': reused_entities if 'reused_entities' in locals() else [],
                'filtered_sentences': [],
                'statistics': {
                    'start_nodes_processed': 0,
                    'raw_count': 0,
                    'filtered_count': 0,
                    'entities_count': 0,
                    'duplicate_avoidance_count': 0,
                    'error': str(e)
                }
            }

    def _apply_advanced_filtering(self, sentences: List[Dict], claim_text: str, 
                                 entities: List[str], max_final_sentences: int,
                                 hop_num: int) -> List[Dict]:
        """
        🔍 Apply Advanced Filtering Pipeline
        """
        print(f"🔍 Hop {hop_num}: Advanced Filtering Pipeline...")
        
        # Build subject keyword set
        subject_keywords = set()
        if self.require_subject_match:
            for ent in entities:
                if any(tok in ent.lower() for tok in ["chim", "bird", "vẹt", "đại bàng", "sếu", "peregrine"]):
                    subject_keywords.add(ent.lower())
            for w in claim_text.lower().split():
                if w.startswith("chim") or w in ["bird", "parrot", "eagle", "sparrow"]:
                    subject_keywords.add(w)

        if self.use_advanced_filtering and self.advanced_filter:
            filtering_results = self.advanced_filter.multi_stage_filtering_pipeline(
                sentences=sentences,
                claim_text=claim_text,
                entities=entities,
                min_quality_score=self.min_quality_score,
                min_relevance_score=self.min_relevance_score,
                min_entity_score=self.min_entity_score,
                stance_delta=self.stance_delta,
                subject_keywords=subject_keywords if self.require_subject_match else None,
                max_final_sentences=max_final_sentences
            )
            
            filtered_sentences = filtering_results['filtered_sentences']
            print(f"✅ Hop {hop_num}: Advanced filtering completed: {len(filtered_sentences)}/{len(sentences)} selected")
            
            # ⚠️ FALLBACK: Nếu không có sentence nào pass filter, giữ lại ít nhất top sentence
            if len(filtered_sentences) == 0 and len(sentences) > 0:
                print(f"⚠️ Hop {hop_num}: No sentences passed filters - keeping top sentence as fallback")
                sorted_sentences = sorted(sentences, key=lambda x: x.get('score', 0), reverse=True)
                filtered_sentences = [sorted_sentences[0]]  # Giữ lại ít nhất 1 sentence
                # Add fallback metadata
                filtered_sentences[0]['filtering_metadata'] = {
                    'is_fallback': True,
                    'reason': 'no_sentences_passed_advanced_filters'
                }
            
        else:
            print(f"⏭️ Hop {hop_num}: Skipped (advanced filtering disabled)")
            filtered_sentences = self._simple_filtering_fallback(sentences, claim_text, max_final_sentences)
        
        return filtered_sentences

    def _aggregate_and_rank_final_results(self, hop_results: List[Dict], 
                                         all_sentences: List[Dict], claim_text: str,
                                         text_graph: TextGraph, max_final_sentences: int) -> Dict:
        """
        🔄 Aggregate and rank final results from all hops
        Note: Duplicate removal đã được thực hiện trong từng hop, nên ở đây chỉ cần aggregate
        """
        print(f"🔄 Aggregating {len(all_sentences)} sentences from {len(hop_results)} hops...")
        
        # ✅ STRONGER DEDUPLICATION: Final check for any remaining duplicates across hops
        seen_sentences = set()
        unique_sentences = []
        
        def normalize_strong(text):
            """Strong normalization for deduplication"""
            return ' '.join(text.replace('_', ' ').strip().lower().split())
        
        for sentence_data in all_sentences:
            sentence_text = sentence_data['sentence']
            normalized_text = normalize_strong(sentence_text)
            
            if normalized_text not in seen_sentences and len(normalized_text) > 10:
                seen_sentences.add(normalized_text)
                # Ensure hop information exists
                if 'multi_hop_metadata' not in sentence_data:
                    sentence_data['multi_hop_metadata'] = {
                        'source_hop': sentence_data.get('hop_number', 1),
                        'aggregation_rank': len(unique_sentences) + 1
                    }
                else:
                    sentence_data['multi_hop_metadata']['aggregation_rank'] = len(unique_sentences) + 1
                unique_sentences.append(sentence_data)
        
        print(f"📊 After final deduplication: {len(unique_sentences)} unique sentences")
        
        # Final SBERT reranking across all hops if available
        if self.use_sbert and self.sbert_model and len(unique_sentences) > 0:
            print("🔄 Final SBERT reranking across all hops...")
            unique_sentences = self._apply_final_sbert_reranking(
                unique_sentences, claim_text,
                use_phobert=False,  # ✅ Dùng SBERT cho final step
                text_graph=text_graph
            )
        
        # Limit to max_final_sentences
        final_sentences = unique_sentences[:max_final_sentences]
        
        # Calculate comprehensive statistics
        comprehensive_stats = self._calculate_multi_hop_statistics(
            hop_results, final_sentences, text_graph, claim_text
        )
        
        # Add detailed hop processing info to stats
        comprehensive_stats['individual_hop_details'] = []
        for hop_result in hop_results:
            if 'individual_results' in hop_result:
                comprehensive_stats['individual_hop_details'].append({
                    'hop_number': hop_result['hop_number'],
                    'individual_processing': hop_result['individual_results']
                })
        
        return {
            'hop_results': hop_results,
            'aggregated_sentences': unique_sentences,
            'final_sentences': final_sentences,
            'comprehensive_statistics': comprehensive_stats,
            'filtering_approach': 'multi_hop_multi_beam_search_individual_loops',
            'num_hops': self.num_hops,
            'total_unique_sentences': len(unique_sentences),
            'final_count': len(final_sentences)
        }

    def _extract_sentences_from_paths(self, paths, text_graph, level: int) -> List[Dict]:
        """Extract sentences from beam search paths"""
        sentences = []
        
        for i, path in enumerate(paths):
            try:
                # Use the correct attributes based on path structure
                node_sequence = path.nodes if hasattr(path, 'nodes') else []
                total_score = path.score if hasattr(path, 'score') else 0.5
                
                if node_sequence and len(node_sequence) > 0:
                    last_node = node_sequence[-1]
                    if last_node in text_graph.graph.nodes:
                        node_data = text_graph.graph.nodes[last_node]
                        
                        # Try multiple ways to identify sentence nodes and extract text
                        sentence_text = None
                        
                        # Method 1: Standard approach
                        if node_data.get('type') == 'sentence' and 'sentence' in node_data:
                            sentence_text = node_data['sentence']
                        
                        # Method 2: Direct text field
                        elif 'text' in node_data:
                            sentence_text = node_data['text']
                        
                        # Method 3: Check if node ID suggests it's a sentence
                        elif last_node.startswith('sentence_'):
                            # Extract sentence by node ID
                            try:
                                sentence_idx = int(last_node.replace('sentence_', ''))
                                # Try to get from text_graph context 
                                all_node_data = list(text_graph.graph.nodes(data=True))
                                for node_id, data in all_node_data:
                                    if (data.get('type') == 'sentence' or 'sentence' in data) and 'sentence' in data:
                                        if node_id == last_node or (hasattr(data, 'index') and data.get('index') == sentence_idx):
                                            sentence_text = data['sentence']
                                            break
                            except:
                                pass
                        
                        # Method 4: Use any text-like field
                        if not sentence_text:
                            for key in ['content', 'value', 'label']:
                                if key in node_data and isinstance(node_data[key], str):
                                    sentence_text = node_data[key]
                                    break
                        
                        if sentence_text and len(sentence_text.strip()) > 5:
                            sentence_info = {
                                'sentence': sentence_text,
                                'score': float(total_score),
                                'level': level,
                                'path_length': len(node_sequence),
                                'source': 'beam_search'
                            }
                            sentences.append(sentence_info)
            except Exception:
                continue
        
        return sentences

    def _apply_keyword_boosting(self, sentence_text: str, claim_text: str, original_score: float) -> float:
        """Apply keyword boosting for sentences containing important claim keywords"""
        boost_factor = 1.0
        
        # Extract key terms from claim
        claim_lower = claim_text.lower()
        sentence_lower = sentence_text.lower()
        
        # Define important keywords to boost
        important_keywords = []
        
        # Extract specific entities/keywords from claim
        claim_words = claim_lower.split()
        for word in claim_words:
            if len(word) > 3 and word not in ['thay', 'cách', 'các', 'để', 'với', 'của', 'trong', 'những', 'được', 'đang', 'là', 'có', 'này', 'cho', 'từ', 'một', 'nhà', 'như']:
                important_keywords.append(word)
        
        # Specific boosting for exact matches
        exact_matches = 0
        for keyword in important_keywords:
            if keyword in sentence_lower:
                exact_matches += 1
                
        # Special boost for very specific keywords
        special_keywords = ['chim', 'bird', 'nghiên_cứu', 'dạy', 'giải_mã', 'tiếng_anh', 'công_nghệ', 'giao_tiếp', 'động_vật']
        special_matches = 0
        for special_kw in special_keywords:
            if special_kw in sentence_lower or special_kw.replace('_', ' ') in sentence_lower:
                special_matches += 1
        
        # Calculate boost
        if exact_matches > 0:
            boost_factor += 0.1 * exact_matches  # 10% boost per exact match
            
        if special_matches > 0:
            boost_factor += 0.2 * special_matches  # 20% boost per special match
            
        # Cap maximum boost
        boost_factor = min(boost_factor, 2.0)  # Max 100% boost
        
        boosted_score = original_score * boost_factor
        
        return boosted_score

    def _get_context_text(self, text_graph: TextGraph) -> str:
        """Get context text from TextGraph"""
        context_sentences = []
        try:
            for _, node_data in text_graph.graph.nodes(data=True):
                if node_data.get('type') == 'sentence' and 'sentence' in node_data:
                    context_sentences.append(node_data['sentence'].replace('_', ' '))
        except Exception:
            pass
        return ' '.join(context_sentences)

    def _simple_filtering_fallback(self, sentences: List[Dict], claim_text: str, 
                                 max_final_sentences: int) -> List[Dict]:
        """Simple filtering fallback when advanced filtering is disabled"""
        if len(sentences) == 0:
            return []
        
        # Sort by score and take top sentences
        sorted_sentences = sorted(sentences, key=lambda x: x.get('score', 0), reverse=True)
        result = sorted_sentences[:max_final_sentences]
        
        # Đảm bảo luôn có ít nhất 1 sentence nếu input có sentences
        if len(result) == 0 and len(sentences) > 0:
            result = [sorted_sentences[0]]
            result[0]['filtering_metadata'] = {
                'is_fallback': True,
                'reason': 'simple_filtering_fallback_minimum_guarantee'
            }
        
        return result

    def _apply_final_sbert_reranking(self, sentences: List[Dict], claim_text: str, use_phobert=False, text_graph=None) -> List[Dict]:
        """Apply final SBERT reranking (with optional PhoBERT support)"""
        if len(sentences) == 0:
            return sentences
            
        # Option 1: Use PhoBERT via TextGraph
        if use_phobert and text_graph and hasattr(text_graph, 'get_sentence_similarity'):
            try:
                print("🔄 Using PhoBERT for final reranking...")
                for sentence_data in sentences:
                    sentence_text = sentence_data['sentence'].replace('_', ' ')
                    similarity = text_graph.get_sentence_similarity(sentence_text, claim_text)
                    sentence_data['final_sbert_score'] = float(similarity)
                    sentence_data['phobert_similarity'] = float(similarity)
                
                # Sort by PhoBERT similarity
                sentences.sort(key=lambda x: x.get('final_sbert_score', 0), reverse=True)
                print(f"✅ PhoBERT reranking completed for {len(sentences)} sentences")
                return sentences
                
            except Exception as e:
                print(f"⚠️ PhoBERT reranking failed, falling back to SBERT: {e}")
        
        # Option 2: Original SBERT approach
        if not self.use_sbert or not self.sbert_model:
            return sentences
        
        try:
            sentence_texts = [s['sentence'].replace('_', ' ') for s in sentences]
            sentence_embeddings = self.sbert_model.encode(sentence_texts)
            claim_embedding = self.sbert_model.encode([claim_text])
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(sentence_embeddings, claim_embedding).flatten()
            
            # Update sentences with SBERT scores
            for i, sentence_data in enumerate(sentences):
                sentence_data['final_sbert_score'] = float(similarities[i])
                sentence_data['sbert_similarity'] = float(similarities[i])
            
            # Sort by SBERT similarity
            sentences.sort(key=lambda x: x.get('final_sbert_score', 0), reverse=True)
            
        except Exception as e:
            print(f"⚠️ SBERT reranking failed: {e}")
        
        return sentences

    def _calculate_multi_hop_statistics(self, hop_results: List[Dict], 
                                       final_sentences: List[Dict], 
                                       text_graph: TextGraph, claim_text: str) -> Dict:
        """Calculate comprehensive statistics across all hops"""
        stats = {
            'multi_hop_summary': {
                'total_hops': len(hop_results),
                'hop_decay_factor': self.hop_decay_factor,
                'total_sentences_across_hops': sum(r['statistics']['raw_count'] for r in hop_results),
                'total_filtered_across_hops': sum(r['statistics']['filtered_count'] for r in hop_results),
                'final_unique_sentences': len(final_sentences)
            },
            'hop_breakdown': {},
            'aggregation_metrics': {}
        }
        
        # Per-hop statistics
        for hop_result in hop_results:
            hop_num = hop_result['hop_number']
            raw_count = hop_result['statistics']['raw_count']
            filtered_count = hop_result['statistics']['filtered_count']
            
            # Safely calculate filtering rate để tránh division by zero
            if raw_count > 0:
                filtering_rate = (1 - filtered_count / raw_count) * 100
            else:
                filtering_rate = 0.0
            
            stats['hop_breakdown'][f'hop_{hop_num}'] = {
                'raw_sentences': raw_count,
                'filtered_sentences': filtered_count,
                'entities_count': hop_result['statistics']['entities_count'],
                'filtering_rate': filtering_rate
            }
        
        # Aggregation metrics
        hop_contributions = {}
        for sentence in final_sentences:
            source_hop = sentence.get('multi_hop_metadata', {}).get('source_hop', 1)
            hop_contributions[f'hop_{source_hop}'] = hop_contributions.get(f'hop_{source_hop}', 0) + 1
        
        # Safely calculate aggregation metrics để tránh division by zero
        num_hops = len(hop_results) if hop_results else 1
        num_final_sentences = len(final_sentences) if final_sentences else 1
        
        stats['aggregation_metrics'] = {
            'hop_contributions': hop_contributions,
            'diversity_score': len(hop_contributions) / num_hops,
            'avg_score': sum(s.get('score', 0) for s in final_sentences) / num_final_sentences if final_sentences else 0,
            'avg_sbert_score': sum(s.get('final_sbert_score', 0) for s in final_sentences) / num_final_sentences if final_sentences else 0
        }
        
        return stats
    
    def _initialize_global_graph(self):
        """Initialize global graph template - NO predefined entities, only OpenAI extraction"""
        # Initialize OpenAI client
        try:
            self.global_graph._init_openai_client()
        except Exception as e:
            print(f"⚠️ OpenAI client initialization failed: {e}")
    
    def _prepare_sample_graph(self, context_sentences, claim_text, claim_sentences):
        """Prepare sample-specific graph - NO predefined entities"""
        # Create new graph without predefined entities
        sample_graph = TextGraph()
        sample_graph._init_openai_client()
        
        # Pass SBERT model to TextGraph for level filtering
        if self.use_sbert and self.sbert_model:
            sample_graph.sbert_model = self.sbert_model
            
        # Set PhoBERT level filtering option (can be toggled)
        sample_graph.use_phobert_level_filtering = getattr(self, 'use_phobert_level_filtering', False)
        
        # Build graph từ VnCoreNLP output  
        sample_graph.build_from_vncorenlp_output(context_sentences, claim_text, claim_sentences)
        
        return sample_graph


def process_sample_with_multi_hop_search(sample_data, model, processor: MultiHopMultiBeamProcessor,
                                        max_levels=3, beam_width_per_level=6, 
                                        max_depth=30, max_final_sentences=30, 
                                        sort_by_original_order=False, **kwargs):
    """
    🚀 Process sample với Multi-Hop Multi-Beam Search
    """
    context = sample_data["context"]
    claim = sample_data["claim"]
    evidence = sample_data["evidence"]
    label = sample_data["label"]
    
    print(f"🔍 Processing claim với Multi-Hop Search: {claim[:100]}...")
    
    try:
        # Setup VnCoreNLP processing (once)
        context_sentences = model.annotate_text(context)
        claim_sentences = model.annotate_text(claim)
        
        total_context_sentences = len(context_sentences)
        # Build or reuse global TextGraph
        if not hasattr(processor, 'global_graph') or processor.global_graph is None:
            processor.global_graph = TextGraph()
            processor._initialize_global_graph()
        
        text_graph = processor._prepare_sample_graph(context_sentences, claim, claim_sentences)
        
        # Use predefined entity pool + optional extraction
        try:
            # DISABLE PREDEFINED ENTITIES - chỉ dùng OpenAI extraction
            predefined_entities = []
            
            # Extract context text for additional entity extraction
            try:
                if isinstance(context_sentences, dict):
                    # VnCoreNLP format: reconstruct sentences from wordForm
                    all_sentence_texts = []
                    for sent_idx, word_list in context_sentences.items():
                        if isinstance(word_list, list):
                            sentence_text = ' '.join([word['wordForm'] for word in word_list if 'wordForm' in word])
                            all_sentence_texts.append(sentence_text)
                    context_text = ' '.join(all_sentence_texts)
                elif isinstance(context_sentences, list) and context_sentences and isinstance(context_sentences[0], dict) and 'sentence' in context_sentences[0]:
                    context_text = ' '.join([sent['sentence'] for sent in context_sentences])
                else:
                    context_text = str(context_sentences)
            except Exception as e:
                print(f"⚠️ Error processing context_sentences: {e}")
                context_text = ""
            
            # 🔗 MULTI-SOURCE ENTITY EXTRACTION & INTEGRATION
            all_entities = []
            
            # 1. Extract phrase entities từ claim
            try:
                claim_phrase_entities = text_graph.extract_phrases_from_claim(
                    claim_text=claim,
                    claim_sentences=claim_sentences
                )
                all_entities.extend(claim_phrase_entities)
                # print(f"🔗 Claim phrases: {len(claim_phrase_entities)} entities")
            except Exception as e:
                print(f"⚠️ Phrase entity extraction failed: {e}")
                claim_phrase_entities = []
            
            # 2. Extract từ OpenAI với dual-source detection
            openai_entities = []
            if context_text and len(context_text) > 50:
                try:
                    openai_entities = text_graph.extract_entities_with_openai(context_text + "\n" + claim)
                    # print(f"🤖 OpenAI entities: {len(openai_entities)} extracted")
                except Exception as e:
                    print(f"⚠️ OpenAI entity extraction failed: {e}")
                    openai_entities = []
            
            # 3. ✅ ENHANCED DUAL-SOURCE ENTITY SCORING & MERGING
            entity_scores = {}  # entity -> {'score': float, 'sources': set, 'original_entity': str}
            dual_source_entities = []
            phrase_only_entities = []
            openai_only_entities = []
            
            # Process phrase entities (from claim)
            for phrase_entity in claim_phrase_entities:
                clean_entity = phrase_entity.strip()
                if len(clean_entity) > 2:
                    normalized = clean_entity.lower().strip()
                    entity_scores[normalized] = {
                        'score': 1.0,  # Equal score for all entities
                        'sources': {'phrase'},
                        'original_entity': clean_entity
                    }
            
            # Process OpenAI entities with overlap detection
            for oa_entity in openai_entities:
                clean_entity = oa_entity.strip()
                if len(clean_entity) <= 2:
                    continue
                    
                normalized = clean_entity.lower().strip()
                
                # Check for overlap with phrase entities
                found_overlap = False
                for phrase_normalized in entity_scores.keys():
                    # Check different overlap patterns
                    if (normalized in phrase_normalized or phrase_normalized in normalized or
                        any(word in phrase_normalized.split() for word in normalized.split() if len(word) > 3)):
                        # DUAL-SOURCE ENTITY DETECTED! 🎯
                        entity_scores[phrase_normalized]['sources'].add('openai')
                        entity_scores[phrase_normalized]['score'] = 1.0  # ✅ EQUAL: All entities same score
                        found_overlap = True
                        break
                
                # If no overlap, add as OpenAI-only entity
                if not found_overlap:
                    entity_scores[normalized] = {
                        'score': 1.0,  # Equal score for all entities
                        'sources': {'openai'},
                        'original_entity': clean_entity
                    }
            
            # Categorize entities by source
            for normalized, info in entity_scores.items():
                sources = info['sources']
                if 'phrase' in sources and 'openai' in sources:
                    dual_source_entities.append(info['original_entity'])
                elif 'phrase' in sources:
                    phrase_only_entities.append(info['original_entity'])
                elif 'openai' in sources:
                    openai_only_entities.append(info['original_entity'])
            
            # Create final entity list (no need to sort by score since all are equal)
            all_entities = [info['original_entity'] for _, info in entity_scores.items()]
            
            # Enhanced logging (minimal)
            # print(f"✅ EQUAL ENTITY SCORING:")
            # print(f"   🎯 Dual-source entities (score=1.0): {len(dual_source_entities)} - {dual_source_entities[:3]}...")
            # print(f"   📝 Phrase-only entities (score=1.0): {len(phrase_only_entities)}")
            # print(f"   🤖 OpenAI-only entities (score=1.0): {len(openai_only_entities)}")
            # print(f"   📊 Total entities: {len(all_entities)} (all entities equal treatment)")
            
            if not all_entities:
                print("⚠️ No entities extracted from any source")
            
            # 4. Add entities to graph với equal scoring system
            if all_entities:
                entity_nodes = text_graph.add_entities_to_graph(
                    entities=all_entities, 
                    context_sentences=context_sentences,
                    entity_scores=entity_scores  # ✅ Pass entity scores (all equal now)
                )
                # Enhanced connection counting với dual-source tracking
                total_connections = sum(len(list(text_graph.graph.neighbors(f"entity_{entity.replace(' ', '_')}"))) 
                                      for entity in all_entities 
                                      if f"entity_{entity.replace(' ', '_')}" in text_graph.graph.nodes)
                
                # Count dual-source vs single-source nodes
                dual_source_nodes = sum(1 for entity in all_entities 
                                      if f"entity_{entity.replace(' ', '_')}" in text_graph.graph.nodes and
                                      text_graph.graph.nodes[f"entity_{entity.replace(' ', '_')}"].get('is_dual_source', False))
                
                # print(f"📊 Enhanced Entity Graph: {len(entity_nodes)} nodes, {total_connections} connections")
                # print(f"   🎯 Dual-source entities: {dual_source_nodes}/{len(entity_nodes)} (prioritized)")
            else:
                entity_nodes = []
                
        except Exception as e:
            print(f"⚠️ Entity setup failed: {e}")
            entity_nodes = []
        
        # 🚀 RUN TRUE MULTI-HOP SEARCH USING NEW PROCESSOR
        enhanced_results = processor.process_multi_hop_search(
            text_graph=text_graph,
            claim_text=claim,
            max_levels=max_levels,
            beam_width_per_level=beam_width_per_level,
            max_depth=max_depth,
            max_final_sentences=max_final_sentences,
            initial_entities=all_entities # Pass extracted entities to the next hop
        )
        
        # Extract final sentences from multi-hop results
        final_sentences = enhanced_results.get('final_sentences', [])
        aggregated_sentences = enhanced_results.get('aggregated_sentences', [])
        
        # ✅ Ensure final_sentences is properly formatted
        if not final_sentences and aggregated_sentences:
            final_sentences = aggregated_sentences[:max_final_sentences]
        
        def normalize_for_dedup(text):
            """Normalize text for deduplication"""
            return ' '.join(text.strip().lower().split())
        
        # Multi-hop results are already processed
        multi_hop_results = {
            'final_sentences': final_sentences,
            'comprehensive_statistics': enhanced_results.get('comprehensive_statistics', {}),
            'filtering_approach': enhanced_results.get('filtering_approach', 'multi_hop_multi_beam_search'),
            'num_hops': enhanced_results.get('num_hops', processor.num_hops)
        }
        comprehensive_stats = multi_hop_results['comprehensive_statistics']
        
        # Process sentences for output
        processed_sentences = []
        for sentence_data in final_sentences:
            sentence_text = sentence_data['sentence'].replace('_', ' ')
            processed_sentences.append({
                "sentence": sentence_text,
                "score": float(sentence_data.get('score', 0)),
                "quality_score": float(sentence_data.get('quality_score', 0)),
                "relevance_score": float(sentence_data.get('relevance_score', 0)),
                "confidence_score": float(sentence_data.get('confidence_score', 0)),
                "sbert_similarity": float(sentence_data.get('sbert_similarity', 0)),
                "final_sbert_score": float(sentence_data.get('final_sbert_score', 0)),
                "level": int(sentence_data.get('level', 0)),
                "source": sentence_data.get('source', 'multi_hop_search'),
                "multi_hop_metadata": sentence_data.get('multi_hop_metadata', {}),
                "filtering_metadata": sentence_data.get('filtering_metadata', {}),
                "quality_analysis": sentence_data.get('quality_analysis', {}),
                "relevance_analysis": sentence_data.get('relevance_analysis', {}),
                "entity_analysis": sentence_data.get('entity_analysis', {}),
                "contradiction_analysis": sentence_data.get('contradiction_analysis', {}),
                "stage_scores": {k: float(v) for k, v in sentence_data.get('stage_scores', {}).items()},
                "stage_metadata": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                                         for kk, vv in md.items()} if isinstance(md, dict) else md
                                   for k, md in sentence_data.get('stage_metadata', {}).items()}
            })
        
        # Coverage statistics
        total_retrieved_sentences = len(processed_sentences)
        coverage_percentage = (total_retrieved_sentences / max(total_context_sentences, 1)) * 100
        
        coverage_stats = {
            "total_context_sentences": total_context_sentences,
            "total_retrieved_sentences": total_retrieved_sentences,
            "coverage_percentage": round(coverage_percentage, 2),
            "max_final_sentences": max_final_sentences,
            "multi_hop_approach": multi_hop_results.get('filtering_approach', 'multi_hop_multi_beam_search'),
            "num_hops": multi_hop_results.get('num_hops', processor.num_hops)
        }
        
        # Overall statistics
        overall_stats = {
            "total_sentences": len(processed_sentences),
            "avg_quality_score": sum(s["quality_score"] for s in processed_sentences) / len(processed_sentences) if processed_sentences else 0,
            "avg_relevance_score": sum(s["relevance_score"] for s in processed_sentences) / len(processed_sentences) if processed_sentences else 0,
            "avg_confidence_score": sum(s["confidence_score"] for s in processed_sentences) / len(processed_sentences) if processed_sentences else 0,
            "avg_final_sbert_score": sum(s["final_sbert_score"] for s in processed_sentences) / len(processed_sentences) if processed_sentences else 0,
            "coverage_statistics": coverage_stats,
            "comprehensive_statistics": comprehensive_stats,
            "config": {
                "max_levels": max_levels,
                "beam_width_per_level": beam_width_per_level,
                "max_depth": max_depth,
                "max_final_sentences": max_final_sentences,
                "num_hops": processor.num_hops,
                "hop_decay_factor": processor.hop_decay_factor,
                "filtering_approach": multi_hop_results.get('filtering_approach', 'multi_hop_multi_beam_search')
            }
        }
        
        # Simple result
        simple_result = {
            "context": context,
            "claim": claim,
            "evidence": evidence,
            "multi_level_evidence": [s["sentence"] for s in processed_sentences],
            "label": label
        }
        
        # Detailed result
        detailed_result = {
            "context": context,
            "claim": claim,
            "evidence": evidence,
            "multi_level_evidence": processed_sentences,
            "statistics": overall_stats,
            "label": label
        }
        
        return simple_result, detailed_result
        
    except Exception as e:
        print(f"❌ Error processing sample with Multi-Hop Search: {e}")
        import traceback
        print(f"🔍 Full error traceback:")
        traceback.print_exc()
        return None, None


def main():
    """
    🚀 Main function cho Multi-Hop Multi-Beam Search Processing
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Hop Multi-Beam Search Fact-Checking")
    parser.add_argument('--input', type=str, default='raw_test.json', 
                       help='Input file path (default: raw_test.json)')
    parser.add_argument('--max_samples', type=int, default=300,
                       help='Maximum number of samples to process (default: 300)')
    parser.add_argument('--output_dir', type=str, default='multi_hop_output',
                       help='Output directory (default: multi_hop_output)')
    
    # Multi-hop specific parameters
    parser.add_argument('--num_hops', type=int, default=3,
                       help='Number of hops in multi-hop search (default: 3)')
    parser.add_argument('--hop_decay_factor', type=float, default=0.8,
                       help='Decay factor for beam width between hops (default: 0.8)')
    
    # Filtering parameters
    parser.add_argument('--use_advanced_filtering', action='store_true', default=True,
                       help='Enable advanced filtering (default: True)')
    parser.add_argument('--use_sbert', action='store_true', default=True,
                       help='Enable SBERT semantic filtering (default: True)')
    parser.add_argument('--use_contradiction_detection', action='store_true', default=True,
                       help='Enable contradiction detection (default: True)')
    parser.add_argument('--use_entity_filtering', action='store_true', default=True,
                       help='Enable entity-based filtering (default: True)')
    parser.add_argument('--min_relevance_score', type=float, default=0.15,
                       help='Minimum relevance score for advanced filtering (default: 0.15)')
    parser.add_argument('--min_quality_score', type=float, default=0.3,
                       help='Minimum quality score for advanced filtering (default: 0.3)')
    parser.add_argument('--max_final_sentences', type=int, default=25,
                       help='Maximum sentences to keep after filtering (default: 25)')
    parser.add_argument('--min_entity_score', type=float, default=0.05,
                       help='Minimum entity relevance score to keep sentence (default: 0.05)')
    parser.add_argument('--stance_delta', type=float, default=0.1,
                       help='SBERT diff threshold for stance detection (default: 0.1)')
    parser.add_argument('--require_subject_match', action='store_true', default=False,
                       help='Keep sentences only if they mention main subject keywords')
    parser.add_argument('--target_ratio', type=float, default=0.5,
                       help='Maximum ratio of extracted sentences vs context (default: 0.5)')
    parser.add_argument('--quiet', action='store_true', default=False,
                       help='Suppress console output (turn off all print messages)')
    parser.add_argument('--sort_by_original_order', action='store_true', default=False,
                       help='Sort final sentences by original order (sentence_0, sentence_1...) instead of scores')
    parser.add_argument('--sort_by_score', action='store_true', default=False,
                       help='Sort final sentences by scores instead of original order')
    parser.add_argument('--use_phobert_level_filtering', action='store_true', default=False,
                       help='Use PhoBERT instead of SBERT for level filtering in beam search')
    
    args = parser.parse_args()

    # 🔇 Quiet mode: suppress all console output if requested
    if getattr(args, 'quiet', False):
        import builtins as _b
        _b.print = lambda *a, **k: None
    
    print("🚀 Starting Multi-Hop Multi-Beam Search Fact-Checking...")
    
    # Initialize Multi-Hop Processor
    processor = MultiHopMultiBeamProcessor(
        num_hops=args.num_hops,
        hop_decay_factor=args.hop_decay_factor,
        use_advanced_filtering=args.use_advanced_filtering,
        use_sbert=args.use_sbert,
        use_contradiction_detection=args.use_contradiction_detection,
        use_entity_filtering=args.use_entity_filtering,
        min_quality_score=args.min_quality_score,
        min_relevance_score=args.min_relevance_score,
        target_ratio=args.target_ratio,
        min_entity_score=args.min_entity_score,
        stance_delta=args.stance_delta,
        require_subject_match=args.require_subject_match,
    )
    
    # Set PhoBERT level filtering option
    processor.use_phobert_level_filtering = args.use_phobert_level_filtering
    if args.use_phobert_level_filtering:
        print("🔍 Using PhoBERT for level filtering in beam search")
    else:
        print("🔍 Using SBERT for level filtering in beam search")
    
    # Setup VnCoreNLP
    original_dir = os.getcwd()
    print(f"📂 Original working directory: {original_dir}")
    
    print("📖 Loading VnCoreNLP model...")
    os.chdir("/Users/nguyennha/Desktop/factchecking/vncorenlp")
    model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], 
                                   save_dir="/Users/nguyennha/Desktop/factchecking/vncorenlp")
    os.chdir(original_dir)
    print(f"📂 Restored working directory: {original_dir}")
    
    # Load dataset
    input_file = os.path.join(original_dir, args.input)
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Configuration - Updated to match new defaults
    num_samples = args.max_samples
    max_levels = 5  # ✅ Updated from 4 to 5
    beam_width_per_level = 15  # ✅ Updated from 10 to 15  
    max_depth = 80  # ✅ Updated from 50 to 80
    max_final_sentences = args.max_final_sentences
    
    print(f"📊 Processing {num_samples} samples with Multi-Hop Multi-Beam Search")
    print(f"⚙️ Parameters: max_levels={max_levels}, beam_width={beam_width_per_level}, max_depth={max_depth}")
    print(f"🔄 Multi-Hop: {args.num_hops} hops, decay_factor={args.hop_decay_factor}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simple_output_file = os.path.join(args.output_dir, f"multi_hop_simple_{timestamp}.json")
    detailed_output_file = os.path.join(args.output_dir, f"multi_hop_detailed_{timestamp}.json")
    
    def append_to_json_array(file_path, new_data):
        """Append data to JSON array file"""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(new_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Process samples
    simple_results = []
    detailed_results = []
    
    for i, sample_data in enumerate(dataset[:num_samples]):
        print(f"\n🔄 Processing sample {i+1}/{num_samples}")
        
        # Determine sorting logic: default to original order unless explicitly requested otherwise
        sort_by_original_order = not args.sort_by_score  # Default True unless --sort_by_score is used
        
        simple_result, detailed_result = process_sample_with_multi_hop_search(
            sample_data, model, processor,
            max_levels=max_levels,
            beam_width_per_level=beam_width_per_level,
            max_depth=max_depth,
            max_final_sentences=max_final_sentences,
            sort_by_original_order=sort_by_original_order
        )
        
        if simple_result and detailed_result:
            simple_results.append(simple_result)
            detailed_results.append(detailed_result)
            
            # Append to output files
            append_to_json_array(simple_output_file, simple_result)
            append_to_json_array(detailed_output_file, detailed_result)
            
            print(f"✅ Sample {i+1} processed successfully")
        else:
            print(f"❌ Sample {i+1} failed")
    
    print(f"\n🎉 Multi-Hop Multi-Beam Search completed!")
    print(f"📊 Processed: {len(simple_results)}/{num_samples} samples")
    print(f"📁 Output files:")
    print(f"   - Simple: {simple_output_file}")
    print(f"   - Detailed: {detailed_output_file}")


if __name__ == "__main__":
    main() 