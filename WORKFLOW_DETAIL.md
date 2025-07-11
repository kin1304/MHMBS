# 📋 LUỒNG CỰC KỲ CHI TIẾT: Process Multi-Hop Multi-Beam Search

## 🚀 TỔNG QUAN LUỒNG XỬ LÝ

Vietnamese Multi-Hop Fact-Checking System thực hiện 8 giai đoạn chính với 32 bước con chi tiết:

```
📊 INPUT → 🔧 PREPROCESSING → 🏗️ GRAPH CONSTRUCTION → 🔗 ENTITY EXTRACTION → 
🔍 ENHANCED SEARCH → 🔍 ADVANCED FILTERING → 🔄 HYBRID RERANKING → 📊 OUTPUT
```

---

## 🔧 GIAI ĐOẠN 1: KHỞI TẠO & CẤU HÌNH (4 bước)

### Bước 1.1: Parse Command Line Arguments
```python
# Các tham số chính được parse:
max_samples: int = 300              # Số sample xử lý
num_hops: int = 3                   # Số hop trong multi-hop search  
max_levels: int = 3                 # Mức độ beam search
beam_width_per_level: int = 6       # Độ rộng beam mỗi level
max_depth: int = 30                 # Độ sâu tối đa search
max_final_sentences: int = 25       # Số câu output cuối cùng

# Thresholds cho filtering
min_quality_score: float = 0.3      # Ngưỡng chất lượng câu
min_relevance_score: float = 0.25   # Ngưỡng liên quan semantic  
min_entity_score: float = 0.05      # Ngưỡng liên quan entity
stance_delta: float = 0.1           # Ngưỡng stance detection

# Feature flags
use_advanced_filtering: bool = True
use_sbert: bool = True
use_entity_filtering: bool = True
sort_by_original_order: bool = False
```

### Bước 1.2: Initialize MultiHopMultiBeamProcessor
```python
processor = MultiHopMultiBeamProcessor(
    num_hops=3,                     # 3 hop reasoning
    hop_decay_factor=0.8,           # Giảm beam width sau mỗi hop
    use_advanced_filtering=True,    # Bật 4-stage filtering
    use_sbert=True,                 # Bật SBERT cho semantic filtering
    use_contradiction_detection=True, # Bật NLI stance detection
    use_entity_filtering=True       # Bật entity-based filtering
)

# Load các models cần thiết:
# ✅ Vietnamese SBERT: "keepitreal/vietnamese-sbert"  
# ✅ XLM-RoBERTa XNLI: "joeddav/xlm-roberta-large-xnli"
# ✅ Advanced Data Filter với multi-stage pipeline
```

### Bước 1.3: Setup VnCoreNLP
```python
# Chuyển working directory để load VnCoreNLP
os.chdir("/Users/nguyennha/Desktop/factchecking/vncorenlp")

# Initialize VnCoreNLP với full annotators
model = py_vncorenlp.VnCoreNLP(
    annotators=["wseg", "pos", "ner", "parse"],  # Word segmentation, POS tagging, NER, Dependency parsing
    save_dir="/Users/nguyennha/Desktop/factchecking/vncorenlp"
)

# Restore original working directory
os.chdir(original_dir)
```

### Bước 1.4: Load Dataset & Create Output Directory
```python
# Load input dataset
with open('raw_test.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "multi_hop_output"
os.makedirs(output_dir, exist_ok=True)

# Prepare output files
simple_output_file = f"multi_hop_output/multi_hop_simple_{timestamp}.json"
detailed_output_file = f"multi_hop_output/multi_hop_detailed_{timestamp}.json"
```

---

## 📄 GIAI ĐOẠN 2: SAMPLE PROCESSING LOOP (3 bước)

### Bước 2.1: Extract Sample Data
```python
for i, sample_data in enumerate(dataset[:max_samples]):
    context = sample_data["context"]      # Đoạn văn context
    claim = sample_data["claim"]          # Câu claim cần fact-check
    evidence = sample_data["evidence"]    # Evidence tham chiếu
    label = sample_data["label"]          # Nhãn SUPPORTS/REFUTES/NEI
```

### Bước 2.2: VnCoreNLP Text Processing
```python
# Process context sentences
context_sentences = model.annotate_text(context)
# Kết quả: Dict[sent_idx, List[token_info]]
# Mỗi token_info có: wordForm, posTag, lemma, index, head, depLabel

# Process claim sentences  
claim_sentences = model.annotate_text(claim)
# Tương tự structure cho claim text

# Calculate total context sentences cho coverage statistics
total_context_sentences = len(context_sentences)
```

### Bước 2.3: TextGraph Construction
```python
# Initialize fresh TextGraph cho mỗi sample
text_graph = TextGraph()
text_graph._init_openai_client()  # Setup OpenAI client cho entity extraction

# Build graph từ VnCoreNLP output
text_graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)

# Graph structure được tạo:
# 📊 Word nodes: với POS filtering (N, Np, V, A, Nc, M, R, P)
# 📊 Sentence nodes: cho mỗi câu trong context
# 📊 Claim node: cho claim text
# 📊 Structural edges: word-to-sentence, word-to-claim connections
# 📊 Dependency edges: dependency parsing relations
```

---

## 🔗 GIAI ĐOẠN 3: MULTI-SOURCE ENTITY EXTRACTION (8 bước)

### Bước 3.1: Phrase-Based Entity Extraction từ Claim
```python
# Extract phrases using VnCoreNLP POS guidance
claim_phrase_entities = text_graph.extract_phrases_from_claim(
    claim_text=claim,
    claim_sentences=claim_sentences
)

# Logic extraction:
# 🔍 Noun sequences: "cựu tổng thống Mỹ", "nghiên cứu khoa học"
# 🔍 Proper nouns: "Donald Trump", "Việt Nam"  
# 🔍 Title patterns: "đầu tiên đối mặt", "chính thức công bố"
# 🔍 Compound phrases: entities connected by dependency relations

print(f"🔗 Claim phrases: {len(claim_phrase_entities)} entities")
```

### Bước 3.2: Context Text Reconstruction
```python
# Reconstruct context text cho OpenAI entity extraction
if isinstance(context_sentences, dict):
    all_sentence_texts = []
    for sent_idx, word_list in context_sentences.items():
        if isinstance(word_list, list):
            sentence_text = ' '.join([word['wordForm'] for word in word_list if 'wordForm' in word])
            all_sentence_texts.append(sentence_text)
    context_text = ' '.join(all_sentence_texts)
```

### Bước 3.3: OpenAI GPT-4o-mini Entity Extraction
```python
if context_text and len(context_text) > 50:
    openai_entities = text_graph.extract_entities_with_openai(context_text + "\n" + claim)
    
    # OpenAI extraction prompt:
    # "Extract important entities (people, places, organizations, events, concepts, dates, quantities) 
    #  from the following Vietnamese text. Return as comma-separated list:"
    
    # Entities được extract: tên người, địa danh, tổ chức, sự kiện, khái niệm, ngày tháng, số lượng
```

### Bước 3.4: Smart Overlap Detection
```python
# Chỉ add OpenAI entities that are NOT already covered by phrases
new_openai_entities = []
for oa_entity in openai_entities:
    is_covered = False
    for phrase in claim_phrase_entities:
        if oa_entity.lower() in phrase.lower() or phrase.lower() in oa_entity.lower():
            is_covered = True
            break
    if not is_covered:
        new_openai_entities.append(oa_entity)

print(f"🤖 OpenAI entities: {len(new_openai_entities)} new entities")
```

### Bước 3.5: Entity Merge & Deduplication
```python
# Combine all entities
all_entities = claim_phrase_entities + new_openai_entities

# Remove duplicates while preserving order
seen = set()
unique_entities = []
for entity in all_entities:
    entity_normalized = entity.lower().strip()
    if entity_normalized not in seen and len(entity.strip()) > 2:
        seen.add(entity_normalized)
        unique_entities.append(entity.strip())

print(f"✅ Total entities: {len(unique_entities)} combined & deduplicated")
```

### Bước 3.6: Add Entities to Graph
```python
if unique_entities:
    entity_nodes = text_graph.add_entities_to_graph(unique_entities, context_sentences)
    
    # Cho mỗi entity:
    # 📊 Tạo entity node: f"entity_{entity.replace(' ', '_')}"
    # 📊 Connect với sentences có chứa entity (fuzzy matching)
    # 📊 Connect với claim node nếu entity xuất hiện trong claim
    # 📊 Calculate connection statistics
    
    total_connections = sum(len(list(text_graph.graph.neighbors(f"entity_{entity.replace(' ', '_')}"))) 
                          for entity in unique_entities 
                          if f"entity_{entity.replace(' ', '_')}" in text_graph.graph.nodes)
    
    print(f"📊 Entity graph: {len(entity_nodes)} nodes, {total_connections} connections")
```

### Bước 3.7: Entity Connection Validation
```python
# Log entities that couldn't be connected
unconnected_entities = []
for entity in unique_entities:
    entity_node_id = f"entity_{entity.replace(' ', '_')}"
    if entity_node_id in text_graph.graph.nodes:
        connections = len(list(text_graph.graph.neighbors(entity_node_id)))
        if connections == 0:
            unconnected_entities.append(entity)

if unconnected_entities:
    print(f"⚠️ Unconnected entities: {len(unconnected_entities)} - {unconnected_entities[:3]}...")
```

### Bước 3.8: Entity Extraction Fallback
```python
# Fallback nếu entity extraction hoàn toàn failed
if not unique_entities:
    print("⚠️ No entities extracted from any source")
    unique_entities = []  # Continue với empty entity list
```

---

## 🔍 GIAI ĐOẠN 4: ENHANCED MULTI-LEVEL BEAM SEARCH (6 bước)

### Bước 4.1: Build Sentence-Claim Semantic Edges
```python
enhanced_results = text_graph.enhanced_multi_level_beam_search_with_direct_connections(
    max_levels=3,
    beam_width_per_level=6,
    max_depth=30,
    min_new_sentences=2,
    claim_text=claim,
    entities=unique_entities,
    filter_top_k=max_final_sentences,
    use_direct_as_starting_points=True,
    sort_by_original_order=sort_by_original_order
)

# Step 1: Build sentence-claim semantic edges
text_graph.build_sentence_claim_semantic_edges(similarity_threshold=0.7)

# PhoBERT similarity computation:
# 🔍 Get sentence embeddings using PhoBERT
# 🔍 Get claim embedding using PhoBERT  
# 🔍 Calculate cosine similarity
# 🔍 Add edge if similarity ≥ 0.7
```

### Bước 4.2: Direct Connection Detection
```python
direct_sentences = text_graph.get_direct_connected_sentences()

# Tìm sentences có direct semantic connection với claim:
# 📊 High-similarity sentences (≥ 0.7 PhoBERT similarity)
# 📊 Immediate evidence candidates
# 📊 Starting points cho multi-hop search

print(f"🎯 Direct connections: {len(direct_sentences)} sentences")
```

### Bước 4.3: Multi-Level Beam Search Execution
```python
# Execute beam search từ direct sentences làm starting points
multi_hop_results = text_graph._beam_search_from_direct_sentences(
    direct_sentences=direct_sentences,
    max_levels=3,
    beam_width_per_level=6,
    max_depth=30,
    min_new_sentences=2,
    claim_text=claim,
    entities=unique_entities,
    filter_top_k=max_final_sentences
)

# Multi-level search structure:
# 📊 Level 0: Direct connected sentences
# 📊 Level 1: 1-hop từ direct sentences
# 📊 Level 2: 2-hop expansion
# 📊 Level 3: 3-hop exploration (if needed)
```

### Bước 4.4: Path Scoring & Ranking
```python
# Mỗi path được score dựa trên:
# 🔢 Word matching score: Overlap giữa claim words và path words
# 🔢 Semantic similarity: PhoBERT similarity scores
# 🔢 Entity bonus: Paths đi qua entities quan trọng
# 🔢 Length penalty: Shorter paths được ưu tiên
# 🔢 Sentence bonus: Paths kết thúc ở sentence nodes
# 🔢 Fuzzy match weight: String similarity

path_score = (word_match_weight * word_overlap + 
              semantic_match_weight * semantic_sim +
              entity_bonus * entity_visits +
              sentence_bonus - 
              length_penalty * path_length)
```

### Bước 4.5: Cross-Level Result Merging
```python
merged_sentences = text_graph._merge_direct_and_multihop_results(
    direct_sentences=direct_sentences,
    multi_hop_results=multi_hop_results,
    sort_by_original_order=sort_by_original_order
)

# Merging logic:
# 🔄 Combine direct + multi-hop results
# 🔄 Advanced deduplication by normalized text
# 🔄 Priority-based sorting (direct sentences trước)
# 🔄 Metadata preservation cho analysis
```

### Bước 4.6: Convert Enhanced Results
```python
# Convert enhanced results sang traditional format
final_sentences = []
seen_texts = set()

for sentence_data in merged_sentences:
    sentence_text = sentence_data.get('sentence_text', '')
    normalized_text = normalize_for_dedup(sentence_text)
    
    if normalized_text not in seen_texts and len(sentence_text.strip()) > 5:
        final_sentences.append({
            'sentence': sentence_text,
            'score': sentence_data.get('path_score', 0.0),
            'level': sentence_data.get('level', 0),
            'source': sentence_data.get('source', 'enhanced_search'),
            'hop_distance': sentence_data.get('hop_distance', 1),
            'similarity_score': sentence_data.get('similarity_score', 0.0),
            'path_length': sentence_data.get('path_length', 1)
        })
        seen_texts.add(normalized_text)
```

---

## 🔍 GIAI ĐOẠN 5: ADVANCED 4-STAGE FILTERING PIPELINE (8 bước)

### Bước 5.1: Stage 1 - Quality Assessment
```python
# Quality filtering với các metrics:
quality_score = (
    length_score * 0.25 +           # Độ dài câu phù hợp (10-200 chars)
    structure_score * 0.25 +        # Cấu trúc câu đầy đủ (có động từ, danh từ)
    content_richness * 0.30 +       # Mật độ thông tin (keywords, entities)
    information_density * 0.20      # Tỷ lệ từ có nghĩa vs stop words
)

# Filter: quality_score ≥ min_quality_score (default: 0.3)
print(f"✅ Quality filtering: {len(quality_filtered)}/{len(sentences)} kept")
```

### Bước 5.2: Stage 2 - Semantic Relevance Filtering
```python
# SBERT semantic similarity với claim
if self.use_sbert and self.sbert_model:
    sentence_embeddings = self.sbert_model.encode([s['sentence'] for s in sentences])
    claim_embedding = self.sbert_model.encode([claim_text])
    
    similarities = cosine_similarity(sentence_embeddings, claim_embedding).flatten()
    
    # Filter: similarity ≥ min_relevance_score (default: 0.25)
    for i, sentence_data in enumerate(sentences):
        sentence_data['relevance_score'] = float(similarities[i])
        if similarities[i] >= min_relevance_score:
            relevance_filtered.append(sentence_data)

print(f"✅ Relevance filtering: {len(relevance_filtered)}/{len(quality_filtered)} kept")
```

### Bước 5.3: Stage 3 - Entity-Based Filtering
```python
# Entity overlap computation
for sentence_data in sentences:
    sentence_text = sentence_data['sentence'].lower()
    
    # Calculate entity overlap score
    matched_entities = 0
    total_entities = len(entities) if entities else 1
    
    for entity in entities:
        # Exact match
        if entity.lower() in sentence_text:
            matched_entities += 1
        # Fuzzy match cho entities dài
        elif len(entity) > 5:
            words = entity.lower().split()
            if any(word in sentence_text for word in words if len(word) > 3):
                matched_entities += 0.5
    
    entity_score = matched_entities / total_entities
    sentence_data['entity_score'] = entity_score
    
    # Filter: entity_score ≥ min_entity_score (default: 0.05)
    if entity_score >= min_entity_score:
        entity_filtered.append(sentence_data)

print(f"✅ Entity filtering: {len(entity_filtered)}/{len(relevance_filtered)} kept")
```

### Bước 5.4: Stage 4 - NLI Stance Detection
```python
if self.use_nli and self.nli_pipeline:
    for sentence_data in sentences:
        sentence_text = sentence_data['sentence']
        
        # XLM-RoBERTa XNLI classification
        # Input format: f"{claim_text} [SEP] {sentence_text}"
        nli_input = f"{claim_text} [SEP] {sentence_text}"
        
        result = self.nli_pipeline(nli_input)
        stance_label = result['label']  # ENTAILMENT/CONTRADICTION/NEUTRAL
        stance_score = result['score']
        
        sentence_data['stance_label'] = stance_label
        sentence_data['stance_score'] = stance_score
        
        # Filter logic:
        # ✅ Keep ENTAILMENT (supporting evidence)
        # ✅ Keep NEUTRAL with high confidence
        # ⚠️ Filter CONTRADICTION with high confidence nếu stance_delta > threshold
        
        if stance_label == 'CONTRADICTION' and stance_score > (0.5 + stance_delta):
            continue  # Filter out contradictory evidence
        else:
            contradiction_filtered.append(sentence_data)

print(f"✅ Contradiction filtering: {len(contradiction_filtered)}/{len(entity_filtered)} kept")
```

### Bước 5.5: Duplicate Removal & Ranking
```python
# Advanced deduplication
seen_sentences = set()
final_sentences = []

for sentence_data in sentences:
    sentence_text = sentence_data['sentence']
    # Strong normalization
    normalized = ' '.join(sentence_text.strip().lower().split())
    
    if normalized not in seen_sentences and len(normalized) > 10:
        seen_sentences.add(normalized)
        
        # Calculate final confidence score
        confidence_score = (
            sentence_data.get('quality_score', 0) * 0.3 +
            sentence_data.get('relevance_score', 0) * 0.3 +
            sentence_data.get('entity_score', 0) * 0.2 +
            sentence_data.get('stance_score', 0) * 0.2
        )
        sentence_data['confidence_score'] = confidence_score
        final_sentences.append(sentence_data)

# Sort by confidence score và limit
final_sentences.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
final_sentences = final_sentences[:max_final_sentences]
```

### Bước 5.6: Fallback Protection
```python
# ⚠️ FALLBACK: Nếu không có sentence nào pass filters
if len(final_sentences) == 0 and len(original_sentences) > 0:
    print(f"⚠️ No sentences passed filters - keeping top sentence as fallback")
    sorted_sentences = sorted(original_sentences, key=lambda x: x.get('score', 0), reverse=True)
    final_sentences = [sorted_sentences[0]]
    final_sentences[0]['filtering_metadata'] = {
        'is_fallback': True,
        'reason': 'no_sentences_passed_advanced_filters'
    }
```

### Bước 5.7: Calculate Filtering Statistics
```python
filtering_statistics = {
    'stage_1_quality': {
        'input': len(original_sentences),
        'output': len(quality_filtered),
        'filtered_out': len(original_sentences) - len(quality_filtered),
        'filtering_rate': (1 - len(quality_filtered)/len(original_sentences)) * 100
    },
    'stage_2_relevance': {
        'input': len(quality_filtered),
        'output': len(relevance_filtered),
        'filtered_out': len(quality_filtered) - len(relevance_filtered),
        'filtering_rate': (1 - len(relevance_filtered)/len(quality_filtered)) * 100 if quality_filtered else 0
    },
    # ... similar cho stages 3 & 4
    'overall': {
        'total_input': len(original_sentences),
        'total_output': len(final_sentences),
        'overall_filtering_rate': (1 - len(final_sentences)/len(original_sentences)) * 100
    }
}
```

### Bước 5.8: Return Filtering Results
```python
return {
    'filtered_sentences': final_sentences,
    'pipeline_results': {
        'input_count': len(original_sentences),
        'final_count': len(final_sentences),
        'stage_results': stage_results,
        'filtering_statistics': filtering_statistics
    }
}
```

---

## 🔄 GIAI ĐOẠN 6: HYBRID RERANKING STRATEGY (4 bước)

### Bước 6.1: Determine Reranking Strategy
```python
# Hybrid strategy:
# 🔄 Hop 1 & 2+: SBERT (fast intermediate reranking)
# 🔄 Final cross-hop: PhoBERT (accurate final reranking)

if hop_level <= 2:
    use_phobert = False  # SBERT cho intermediate hops
else:
    use_phobert = True   # PhoBERT cho final cross-hop ranking
```

### Bước 6.2: SBERT Intermediate Reranking
```python
if not use_phobert and self.use_sbert and self.sbert_model:
    try:
        sentence_texts = [s['sentence'].replace('_', ' ') for s in sentences]
        sentence_embeddings = self.sbert_model.encode(sentence_texts)
        claim_embedding = self.sbert_model.encode([claim_text])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(sentence_embeddings, claim_embedding).flatten()
        
        # Update sentences với SBERT scores
        for i, sentence_data in enumerate(sentences):
            sentence_data['final_sbert_score'] = float(similarities[i])
            sentence_data['sbert_similarity'] = float(similarities[i])
        
        # Sort by SBERT similarity
        sentences.sort(key=lambda x: x.get('final_sbert_score', 0), reverse=True)
        
        print(f"✅ SBERT intermediate reranking completed for {len(sentences)} sentences")
    except Exception as e:
        print(f"⚠️ SBERT reranking failed: {e}")
```

### Bước 6.3: PhoBERT Final Reranking
```python
if use_phobert and text_graph and hasattr(text_graph, 'get_sentence_similarity'):
    try:
        print("🔄 Using PhoBERT for final reranking...")
        for sentence_data in sentences:
            sentence_text = sentence_data['sentence'].replace('_', ' ')
            
            # Use TextGraph's PhoBERT similarity method
            similarity = text_graph.get_sentence_similarity(sentence_text, claim_text)
            
            sentence_data['final_sbert_score'] = float(similarity)
            sentence_data['phobert_similarity'] = float(similarity)
        
        # Sort by PhoBERT similarity
        sentences.sort(key=lambda x: x.get('final_sbert_score', 0), reverse=True)
        
        print(f"✅ PhoBERT final reranking completed for {len(sentences)} sentences")
    except Exception as e:
        print(f"⚠️ PhoBERT reranking failed, falling back to SBERT: {e}")
```

### Bước 6.4: Reranking Result Validation
```python
# Ensure all sentences have reranking scores
for sentence_data in sentences:
    if 'final_sbert_score' not in sentence_data:
        sentence_data['final_sbert_score'] = sentence_data.get('score', 0.5)
    
    # Backup similarity score
    if 'sbert_similarity' not in sentence_data and 'phobert_similarity' not in sentence_data:
        sentence_data['sbert_similarity'] = sentence_data.get('final_sbert_score', 0.5)
```

---

## 📊 GIAI ĐOẠN 7: RESULT PROCESSING & STATISTICS (5 bước)

### Bước 7.1: Process Sentences for Output Format
```python
processed_sentences = []
for sentence_data in final_sentences:
    sentence_text = sentence_data['sentence'].replace('_', ' ')
    
    processed_sentence = {
        "sentence": sentence_text,
        "score": float(sentence_data.get('score', 0)),
        "quality_score": float(sentence_data.get('quality_score', 0)),
        "relevance_score": float(sentence_data.get('relevance_score', 0)),
        "confidence_score": float(sentence_data.get('confidence_score', 0)),
        "sbert_similarity": float(sentence_data.get('sbert_similarity', 0)),
        "final_sbert_score": float(sentence_data.get('final_sbert_score', 0)),
        "level": int(sentence_data.get('level', 0)),
        "source": sentence_data.get('source', 'multi_hop_search'),
        
        # Detailed metadata
        "multi_hop_metadata": sentence_data.get('multi_hop_metadata', {}),
        "filtering_metadata": sentence_data.get('filtering_metadata', {}),
        "quality_analysis": sentence_data.get('quality_analysis', {}),
        "relevance_analysis": sentence_data.get('relevance_analysis', {}),
        "entity_analysis": sentence_data.get('entity_analysis', {}),
        "contradiction_analysis": sentence_data.get('contradiction_analysis', {}),
        
        # Stage scores (convert to float)
        "stage_scores": {k: float(v) for k, v in sentence_data.get('stage_scores', {}).items()},
        "stage_metadata": {
            k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                for kk, vv in md.items()} if isinstance(md, dict) else md
            for k, md in sentence_data.get('stage_metadata', {}).items()
        }
    }
    processed_sentences.append(processed_sentence)
```

### Bước 7.2: Calculate Coverage Statistics
```python
total_retrieved_sentences = len(processed_sentences)
coverage_percentage = (total_retrieved_sentences / max(total_context_sentences, 1)) * 100

coverage_stats = {
    "total_context_sentences": total_context_sentences,
    "total_retrieved_sentences": total_retrieved_sentences,
    "coverage_percentage": round(coverage_percentage, 2),
    "max_final_sentences": max_final_sentences,
    "multi_hop_approach": enhanced_results.get('filtering_approach', 'enhanced_multi_level_beam_search'),
    "num_hops": 3
}
```

### Bước 7.3: Calculate Overall Statistics
```python
overall_stats = {
    "total_sentences": len(processed_sentences),
    "avg_quality_score": sum(s["quality_score"] for s in processed_sentences) / len(processed_sentences) if processed_sentences else 0,
    "avg_relevance_score": sum(s["relevance_score"] for s in processed_sentences) / len(processed_sentences) if processed_sentences else 0,
    "avg_confidence_score": sum(s["confidence_score"] for s in processed_sentences) / len(processed_sentences) if processed_sentences else 0,
    "avg_final_sbert_score": sum(s["final_sbert_score"] for s in processed_sentences) / len(processed_sentences) if processed_sentences else 0,
    "coverage_statistics": coverage_stats,
    "comprehensive_statistics": enhanced_results.get('statistics', {}),
    "config": {
        "max_levels": max_levels,
        "beam_width_per_level": beam_width_per_level,
        "max_depth": max_depth,
        "max_final_sentences": max_final_sentences,
        "num_hops": 3,
        "filtering_approach": enhanced_results.get('filtering_approach', 'enhanced_multi_level_beam_search')
    }
}
```

### Bước 7.4: Create Simple Result
```python
simple_result = {
    "context": context,
    "claim": claim,
    "evidence": evidence,
    "multi_level_evidence": [s["sentence"] for s in processed_sentences],  # Chỉ sentence text
    "label": label
}
```

### Bước 7.5: Create Detailed Result
```python
detailed_result = {
    "context": context,
    "claim": claim,
    "evidence": evidence,
    "multi_level_evidence": processed_sentences,  # Full sentence objects với metadata
    "statistics": overall_stats,
    "label": label
}
```

---

## 📁 GIAI ĐOẠN 8: OUTPUT & EXPORT (3 bước)

### Bước 8.1: Append Results to JSON Arrays
```python
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

# Append current sample results
append_to_json_array(simple_output_file, simple_result)
append_to_json_array(detailed_output_file, detailed_result)
```

### Bước 8.2: Track Processing Progress
```python
print(f"✅ Sample {i+1}/{num_samples} processed successfully")
print(f"   📊 Found {len(processed_sentences)} evidence sentences")
print(f"   📈 Coverage: {coverage_percentage:.1f}%")
print(f"   🎯 Avg Quality: {overall_stats['avg_quality_score']:.3f}")
print(f"   🔗 Avg Relevance: {overall_stats['avg_relevance_score']:.3f}")
```

### Bước 8.3: Final Summary Report
```python
print(f"\n🎉 Multi-Hop Multi-Beam Search completed!")
print(f"📊 Processed: {len(simple_results)}/{num_samples} samples")
print(f"📁 Output files:")
print(f"   - Simple: {simple_output_file}")
print(f"   - Detailed: {detailed_output_file}")

# Final statistics across all samples
total_evidence_found = sum(len(result['multi_level_evidence']) for result in simple_results)
avg_evidence_per_sample = total_evidence_found / len(simple_results) if simple_results else 0

print(f"📈 Final Statistics:")
print(f"   - Total evidence sentences found: {total_evidence_found}")
print(f"   - Average evidence per sample: {avg_evidence_per_sample:.1f}")
print(f"   - Success rate: {len(simple_results)/num_samples*100:.1f}%")
```

---

## 🔧 CÁC THAM SỐ QUAN TRỌNG

### Search Parameters
```python
max_levels = 3              # Multi-hop levels (optimized từ 5 → 3)
beam_width_per_level = 6    # Beam width (optimized từ 20 → 6)
max_depth = 30              # Search depth (optimized từ 100 → 30)
max_final_sentences = 25    # Output limit
```

### Filtering Thresholds
```python
similarity_threshold = 0.7   # Sentence-claim semantic similarity
min_quality_score = 0.3     # Quality assessment threshold
min_relevance_score = 0.25  # SBERT relevance threshold
min_entity_score = 0.05     # Entity overlap threshold
stance_delta = 0.1          # NLI stance detection delta
```

### Model Configurations
```python
SBERT_MODEL = "keepitreal/vietnamese-sbert"           # Vietnamese SBERT
PHOBERT_MODEL = "vinai/phobert-base"                  # Vietnamese PhoBERT
NLI_MODEL = "joeddav/xlm-roberta-large-xnli"        # Multilingual NLI
OPENAI_MODEL = "gpt-4o-mini"                         # Entity extraction
```

---

## 📊 OUTPUT FORMATS

### Simple Result Format
```json
{
  "context": "Original context text...",
  "claim": "Claim to fact-check...",
  "evidence": "Reference evidence...",
  "multi_level_evidence": [
    "Evidence sentence 1...",
    "Evidence sentence 2...",
    "Evidence sentence 3..."
  ],
  "label": "SUPPORTS/REFUTES/NEI"
}
```

### Detailed Result Format  
```json
{
  "context": "Original context text...",
  "claim": "Claim to fact-check...",
  "evidence": "Reference evidence...",
  "multi_level_evidence": [
    {
      "sentence": "Evidence sentence...",
      "score": 0.85,
      "quality_score": 0.72,
      "relevance_score": 0.68,
      "confidence_score": 0.74,
      "sbert_similarity": 0.71,
      "final_sbert_score": 0.78,
      "level": 1,
      "source": "enhanced_search",
      "multi_hop_metadata": {...},
      "filtering_metadata": {...},
      "stage_scores": {...}
    }
  ],
  "statistics": {
    "total_sentences": 15,
    "avg_quality_score": 0.68,
    "avg_relevance_score": 0.61,
    "coverage_statistics": {...},
    "comprehensive_statistics": {...}
  },
  "label": "SUPPORTS/REFUTES/NEI"
}
```

---

## ⚡ PERFORMANCE OPTIMIZATIONS

### Memory Optimizations
- ✅ Reuse VnCoreNLP model across samples
- ✅ Cache entity extraction results
- ✅ Stream JSON output instead of loading all in memory
- ✅ Clear intermediate variables after processing

### Speed Optimizations  
- ✅ Reduced search parameters (3 levels, beam width 6, depth 30)
- ✅ Parallel entity extraction (phrase + OpenAI)
- ✅ Hybrid reranking (SBERT intermediate, PhoBERT final)
- ✅ Early termination for low-quality paths

### Quality Optimizations
- ✅ Multi-source entity extraction with smart deduplication
- ✅ 4-stage filtering pipeline with fallback protection
- ✅ Advanced deduplication with strong text normalization
- ✅ Comprehensive statistics tracking for analysis

---

## 🚀 EXECUTION EXAMPLE

```bash
# Basic execution
python process_multi_hop_multi_beam_search.py --input raw_test.json --max_samples 10

# Advanced configuration
python process_multi_hop_multi_beam_search.py \
    --input raw_test.json \
    --max_samples 50 \
    --num_hops 3 \
    --max_final_sentences 20 \
    --min_quality_score 0.4 \
    --min_relevance_score 0.3 \
    --use_advanced_filtering \
    --use_sbert \
    --use_entity_filtering \
    --sort_by_original_order

# Entity-focused search
python process_multi_hop_multi_beam_search.py \
    --input raw_test.json \
    --max_samples 20 \
    --min_entity_score 0.1 \
    --use_entity_filtering \
    --require_subject_match

# Filtering comparison
python process_multi_hop_multi_beam_search.py \
    --input raw_test.json \
    --max_samples 30 \
    --min_quality_score 0.5 \
    --min_relevance_score 0.0 \
    --min_entity_score 0.0
```

Luồng xử lý này được thiết kế để cung cấp comprehensive fact-checking với high precision cho tiếng Việt thông qua multi-hop reasoning và advanced filtering strategies. 