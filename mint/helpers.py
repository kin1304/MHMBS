#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MINT TextGraph Helper Functions
Helper functions for CLI and text processing
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import py_vncorenlp
from dotenv import load_dotenv
from .text_graph import TextGraph

def detect_device():
    """Automatically detect and configure optimal device (GPU/CPU)"""
    try:
        import torch
        
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            device_info = {
                'type': 'GPU',
                'name': gpu_name,
                'memory_gb': f"{gpu_memory:.1f}GB",
                'device': device,
                'use_gpu_optimizations': True
            }
        else:
            device = 'cpu'
            cpu_count = torch.get_num_threads()
            device_info = {
                'type': 'CPU',
                'name': f'{cpu_count} threads',
                'memory_gb': 'N/A',
                'device': device,
                'use_gpu_optimizations': False
            }
            
    except ImportError:
        # Fallback nếu torch không có
        device_info = {
            'type': 'CPU',
            'name': 'Unknown (torch not available)',
            'memory_gb': 'N/A', 
            'device': 'cpu',
            'use_gpu_optimizations': False
        }
    
    return device_info

def get_optimized_config_for_device(device_info, base_config):
    """Get optimized configuration based on detected device (PCA removed)"""
    if device_info['use_gpu_optimizations']:
        # GPU optimizations - sử dụng full embeddings với FAISS
        return {
            'similarity_threshold': base_config.get('similarity_threshold', 0.85),
            'top_k': base_config.get('top_k', 5),
            'use_faiss': base_config.get('use_faiss', True)
        }
    else:
        # CPU optimizations - giảm top_k, tăng threshold, có thể tắt FAISS
        return {
            'similarity_threshold': base_config.get('cpu_similarity_threshold', 0.9),
            'top_k': base_config.get('cpu_top_k', 3),
            'use_faiss': base_config.get('cpu_use_faiss', False)  # FAISS có thể problematic trên một số CPU
        }

def load_config():
    """Load configuration from .env file with defaults"""
    load_dotenv()
    
    config = {
        # Semantic Similarity defaults (PCA removed)
        'similarity_threshold': float(os.getenv('DEFAULT_SIMILARITY_THRESHOLD', '0.85')),
        'top_k': int(os.getenv('DEFAULT_TOP_K', '5')),
        'use_faiss': os.getenv('DEFAULT_USE_FAISS', 'true').lower() == 'true',
        
        # OpenAI defaults
        'openai_model': os.getenv('DEFAULT_OPENAI_MODEL', 'gpt-4o-mini'),
        'openai_temperature': float(os.getenv('DEFAULT_OPENAI_TEMPERATURE', '0.0')),
        'openai_max_tokens': int(os.getenv('DEFAULT_OPENAI_MAX_TOKENS', '1000')),
        
        # Visualization defaults
        'figure_width': float(os.getenv('DEFAULT_FIGURE_WIDTH', '15')),
        'figure_height': float(os.getenv('DEFAULT_FIGURE_HEIGHT', '10')),
        'dpi': int(os.getenv('DEFAULT_DPI', '300')),
        
        # System defaults
        'vncorenlp_path': os.getenv('DEFAULT_VNCORENLP_PATH', 'vncorenlp'),
        'export_graph': os.getenv('DEFAULT_EXPORT_GRAPH', 'text_graph.gexf'),
        'enable_statistics': os.getenv('DEFAULT_ENABLE_STATISTICS', 'true').lower() == 'true',
        'enable_visualization': os.getenv('DEFAULT_ENABLE_VISUALIZATION', 'true').lower() == 'true',
        
        # Auto-save defaults
        'auto_save_graph': os.getenv('DEFAULT_AUTO_SAVE_GRAPH', 'true').lower() == 'true',
        'auto_save_path': os.getenv('DEFAULT_AUTO_SAVE_PATH', 'output/graph_auto_{timestamp}.gexf'),
        
        # Beam search defaults
        'enable_beam_search': os.getenv('DEFAULT_ENABLE_BEAM_SEARCH', 'false').lower() == 'true',
        'beam_width': int(os.getenv('DEFAULT_BEAM_WIDTH', '10')),
        'beam_max_depth': int(os.getenv('DEFAULT_BEAM_MAX_DEPTH', '6')),
        'beam_max_paths': int(os.getenv('DEFAULT_BEAM_MAX_PATHS', '20')),
        'beam_export_dir': os.getenv('DEFAULT_BEAM_EXPORT_DIR', 'output'),
        
        # Demo data
        'demo_data_path': os.getenv('DEMO_DATA_PATH', 'data/demo.json'),
        
        # CPU mode (low performance fallback - no PCA)
        'cpu_similarity_threshold': float(os.getenv('CPU_SIMILARITY_THRESHOLD', '0.9')),
        'cpu_top_k': int(os.getenv('CPU_TOP_K', '3')),
        'cpu_use_faiss': os.getenv('CPU_USE_FAISS', 'false').lower() == 'true',
    }
    
    return config

def load_demo_data():
    """Load demo data from JSON file"""
    config = load_config()
    demo_path = config['demo_data_path']
    
    # Fallback SAWACO data if file not found
    fallback_data = {
        "context": """(PLO)- Theo Tổng Công ty Cấp nước Sài Gòn (SAWACO) việc cúp nước là để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp. SAWACO cho biết đây là phương án để đảm bảo cung cấp nước sạch an toàn, liên tục phục vụ cho người dân TP. Vì vậy, SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác nêu trên. Thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật). Các khu vực tạm ngưng cung cấp nước gồm quận 6, 8, 12, Gò Vấp, Tân Bình, Tân Phú, Bình Tân và huyện Hóc Môn, Bình Chánh.""",
        "claim": """SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)."""
    }
    
    try:
        if os.path.exists(demo_path):
            with open(demo_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('context', ''), data.get('claim', '')
        else:
            print(f"⚠️ Demo file not found: {demo_path}, using fallback SAWACO data")
            return fallback_data["context"], fallback_data["claim"]
    except Exception as e:
        print(f"⚠️ Error loading demo data: {e}, using fallback")
        return fallback_data["context"], fallback_data["claim"]

def apply_device_optimizations(args, device_info, verbose=False):
    """Apply device-specific optimizations to arguments (PCA removed)"""
    config = load_config()
    optimized_config = get_optimized_config_for_device(device_info, config)
    
    # Only override if user didn't specify custom values
    if not getattr(args, 'similarity_threshold_overridden', False):
        args.similarity_threshold = optimized_config['similarity_threshold']
    if not getattr(args, 'top_k_overridden', False):
        args.top_k = optimized_config['top_k']
    
    # Set technical flags
    args.disable_faiss = not optimized_config['use_faiss']
    
    if verbose:
        optimization_type = "GPU" if device_info['use_gpu_optimizations'] else "CPU"
        print(f"🔧 {optimization_type} optimizations applied (full embeddings - no PCA):")
        print(f"  Similarity threshold: {args.similarity_threshold}")
        print(f"  Top-K: {args.top_k}")
        print(f"  Use FAISS: {not args.disable_faiss}")
        print(f"  Embedding dimensions: 768 (full PhoBERT)")

def validate_inputs(args):
    """Validate and extract input data from arguments"""
    context = None
    claim = None
    
    if args.demo:
        context, claim = load_demo_data()
        if args.verbose:
            demo_name = "Bánh cuốn Thụy Khuê" if "bánh cuốn" in context.lower() else "SAWACO"
            print(f"📋 Using demo data ({demo_name} example)")
    
    elif args.input_file:
        if not os.path.exists(args.input_file):
            raise ValueError(f"Input file not found: {args.input_file}")
        
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Xử lý cả array và object đơn
            if isinstance(data, list):
                if len(data) == 0:
                    raise ValueError("Input file contains empty array")
                # Lấy sample đầu tiên nếu là array
                sample = data[0]
                context = sample.get('context', '')
                claim = sample.get('claim', '')
                if args.verbose:
                    print(f"📋 Using first sample from {len(data)} samples in input file")
            else:
                # Xử lý object đơn
                context = data.get('context', '')
                claim = data.get('claim', '')
            
            if not context or not claim:
                raise ValueError("Input data must contain 'context' and 'claim' fields")
                
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {args.input_file}")
    
    elif args.context and args.claim:
        context = args.context
        claim = args.claim
    
    else:
        raise ValueError("Must provide either --demo, --input-file, or both --context and --claim")
    
    if not context.strip() or not claim.strip():
        raise ValueError("Context and claim cannot be empty")
    
    # Auto-detect device and apply optimizations
    device_info = detect_device()
    apply_device_optimizations(args, device_info, args.verbose)
    
    return context.strip(), claim.strip(), device_info

def setup_vncorenlp(vncorenlp_path, verbose=False, auto_download=True):
    """Setup VnCoreNLP model with automatic download if needed"""
    
    # Convert to absolute path
    if not os.path.isabs(vncorenlp_path):
        vncorenlp_path = os.path.abspath(vncorenlp_path)
    
    # Check if VnCoreNLP exists
    jar_path = os.path.join(vncorenlp_path, "VnCoreNLP-1.2.jar")
    models_path = os.path.join(vncorenlp_path, "models")
    
    if not (os.path.exists(jar_path) and os.path.exists(models_path)):
        if auto_download:
            if verbose:
                print(f"  VnCoreNLP not found at: {vncorenlp_path}")
                print(f"  Auto-downloading VnCoreNLP...")
            try:
                vncorenlp_path = download_vncorenlp(vncorenlp_path, verbose)
            except Exception as e:
                raise RuntimeError(f"Failed to auto-download VnCoreNLP: {e}")
        else:
            raise RuntimeError(f"VnCoreNLP not found at: {vncorenlp_path}")
    
    try:
        if verbose:
            print(f"  Loading VnCoreNLP from: {vncorenlp_path}")
        
        model = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_path)
        
        if verbose:
            print("  ✅ VnCoreNLP loaded successfully")
        
        return model
    
    except Exception as e:
        raise RuntimeError(f"Failed to load VnCoreNLP: {e}")

def process_text_data(model, context, claim, verbose=False):
    """Process context and claim with VnCoreNLP"""
    try:
        if verbose:
            print("  Processing context...")
        context_sentences = model.annotate_text(context)
        
        if verbose:
            print("  Processing claim...")
        claim_sentences = model.annotate_text(claim)
        
        if verbose:
            print(f"  ✅ Processed {len(context_sentences)} context sentences")
            print(f"  ✅ Processed {len(claim_sentences)} claim sentences")
        
        return context_sentences, claim_sentences
    
    except Exception as e:
        raise RuntimeError(f"Failed to process text data: {e}")

def configure_textgraph_parameters(text_graph, args):
    """Configure TextGraph parameters from arguments with .env defaults (PCA removed)"""
    config = load_config()
    
    # Use CLI args if provided, otherwise use .env defaults
    text_graph.similarity_threshold = getattr(args, 'similarity_threshold', config['similarity_threshold'])
    text_graph.top_k_similar = getattr(args, 'top_k', config['top_k'])
    
    # OpenAI parameters
    if hasattr(text_graph, '_update_openai_model'):
        text_graph._update_openai_model(
            model=getattr(args, 'openai_model', config['openai_model']),
            temperature=getattr(args, 'openai_temperature', config['openai_temperature']),
            max_tokens=getattr(args, 'openai_max_tokens', config['openai_max_tokens'])
        )

def build_complete_graph(context, claim, context_sentences, claim_sentences, args):
    """Build complete text graph with all features"""
    # Initialize TextGraph
    text_graph = TextGraph()
    
    # Configure parameters
    configure_textgraph_parameters(text_graph, args)
    
    # Configure POS filtering (mặc định bật, có thể tắt bằng --disable-pos-filtering)
    if getattr(args, 'disable_pos_filtering', False):
        text_graph.set_pos_filtering(enable=False)
        if args.verbose:
            print("  ⚠️ POS filtering disabled - all words will be included")
    else:
        # POS filtering được bật mặc định, có thể tùy chỉnh tags
        custom_pos_tags = None
        if hasattr(args, 'pos_tags') and args.pos_tags:
            custom_pos_tags = [tag.strip() for tag in args.pos_tags.split(',')]
        text_graph.set_pos_filtering(enable=True, custom_pos_tags=custom_pos_tags)
        if args.verbose:
            print(f"  ✅ POS filtering enabled. Using tags: {text_graph.important_pos_tags}")
    
    # Build basic graph
    if args.verbose:
        print("  Building basic graph structure...")
    text_graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)
    
    # Entity extraction
    if not args.disable_entities:
        if args.verbose:
            print("  Extracting entities with OpenAI...")
        try:
            entity_nodes = text_graph.extract_and_add_entities(context, context_sentences)
            if args.verbose:
                print(f"  ✅ Added {len(entity_nodes)} entity nodes")
        except Exception as e:
            if args.verbose:
                print(f"  ⚠️ Entity extraction failed: {e}")
    
    # Semantic similarity (without PCA)
    if not args.disable_semantic:
        if args.verbose:
            print("  Building semantic similarity edges (full embeddings - no PCA)...")
        try:
            use_faiss = not args.disable_faiss
            
            edges_added = text_graph.build_semantic_similarity_edges(
                use_faiss=use_faiss
            )
            if args.verbose:
                print(f"  ✅ Added {edges_added} semantic edges")
        except Exception as e:
            if args.verbose:
                print(f"  ⚠️ Semantic similarity failed: {e}")
    
    # Auto-save graph if enabled
    if getattr(args, 'auto_save_graph', True):
        try:
            auto_save_path = getattr(args, 'auto_save_path', 'output/graph_auto_{timestamp}.gexf')
            saved_path = auto_save_graph(text_graph, auto_save_path, args.verbose)
            if args.verbose:
                print(f"  💾 Graph auto-saved to: {saved_path}")
        except Exception as e:
            if args.verbose:
                print(f"  ⚠️ Auto-save failed: {e}")
    
    # Beam Search for path finding (optional)
    if getattr(args, 'beam_search', False):
        if args.verbose:
            print("  🎯 Running beam search to find paths from claim to sentences...")
        try:
            beam_width = getattr(args, 'beam_width', 10)
            max_depth = getattr(args, 'beam_max_depth', 6)
            max_paths = getattr(args, 'beam_max_paths', 20)
            export_dir = getattr(args, 'beam_export_dir', 'output')
            
            # Run beam search
            paths = text_graph.beam_search_paths(
                beam_width=beam_width,
                max_depth=max_depth,
                max_paths=max_paths
            )
            
            if paths:
                # Export results
                json_file, summary_file = text_graph.export_beam_search_results(
                    paths, 
                    output_dir=export_dir
                )
                
                # Print statistics
                if args.verbose:
                    stats = text_graph.analyze_paths_quality(paths)
                    print(f"  📊 Beam search results:")
                    print(f"    Found {stats['total_paths']} paths")
                    print(f"    Avg score: {stats['avg_score']:.3f}")
                    print(f"    Avg length: {stats['avg_length']:.1f} nodes")
                    print(f"    Paths to sentences: {stats['paths_to_sentences']}")
                    print(f"    Paths through entities: {stats['paths_through_entities']}")
                    print(f"    Files saved: {json_file}, {summary_file}")
            else:
                if args.verbose:
                    print("  ⚠️ No paths found from claim to sentences")
                    
        except Exception as e:
            if args.verbose:
                print(f"  ⚠️ Beam search failed: {e}")
    
    return text_graph

def print_statistics(text_graph, verbose=False):
    """Print detailed statistics"""
    print("\n" + "="*50)
    print("📊 DETAILED STATISTICS")
    print("="*50)
    
    stats = text_graph.get_detailed_statistics()
    
    # Basic statistics
    print(f"📈 Graph Overview:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"    - Structural edges: {stats['structural_edges']}")
    print(f"    - Dependency edges: {stats['dependency_edges']}")
    print(f"    - Entity edges: {stats.get('entity_structural_edges', 0)}")
    print(f"    - Semantic edges: {stats.get('semantic_edges', 0)}")
    
    print(f"\n📍 Node Types:")
    print(f"  Word nodes: {stats['word_nodes']}")
    print(f"  Sentence nodes: {stats['sentence_nodes']}")
    print(f"  Claim nodes: {stats['claim_nodes']}")
    print(f"  Entity nodes: {stats.get('entity_nodes', 0)}")
    
    # Word analysis
    print(f"\n📝 Text Analysis:")
    print(f"  Unique words: {stats['unique_words']}")
    print(f"  Shared words (context & claim): {stats['shared_words_count']}")
    print(f"  Average words per sentence: {stats['average_words_per_sentence']:.1f}")
    
    # Entity information
    if stats.get('entities'):
        print(f"\n🏷️ Entities Extracted:")
        for entity in stats['entities'][:10]:  # Show first 10
            print(f"  '{entity['name']}' - {entity['connected_sentences']} connections")
        if len(stats['entities']) > 10:
            print(f"  ... and {len(stats['entities']) - 10} more entities")
    
    # Semantic similarity info
    if stats.get('semantic_edges', 0) > 0:
        semantic_stats = stats['semantic_statistics']
        print(f"\n🔗 Semantic Similarity:")
        print(f"  Total semantic edges: {semantic_stats['total_semantic_edges']}")
        print(f"  Average similarity: {semantic_stats['average_similarity']:.3f}")
        print(f"  Similarity range: {semantic_stats['min_similarity']:.3f} - {semantic_stats['max_similarity']:.3f}")
        
        print("  Similarity distribution:")
        for range_key, count in semantic_stats['similarity_distribution'].items():
            if count > 0:
                print(f"    {range_key}: {count} edges")
    
    # Dependency statistics
    if verbose:
        dep_stats = stats['dependency_statistics']
        print(f"\n🔗 Dependency Parsing:")
        print(f"  Total dependency edges: {dep_stats['total_dependency_edges']}")
        print(f"  Dependency types: {len(dep_stats['dependency_types'])}")
        
        print("  Most common dependencies:")
        for dep_type, count in dep_stats['most_common_dependencies'][:8]:
            print(f"    '{dep_type}': {count} edges")
        
        print(f"\n📊 Most Frequent Words:")
        for word, freq in stats['most_frequent_words']:
            print(f"  '{word}': {freq} times")

def auto_save_graph(text_graph, path_pattern, verbose=False):
    """
    Tự động lưu graph với timestamp và tạo thư mục nếu cần
    
    Args:
        text_graph: TextGraph object to save
        path_pattern: Path pattern with {timestamp} placeholder
        verbose: Whether to print verbose output
        
    Returns:
        str: Actual saved file path
    """
    import datetime
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Replace {timestamp} in path
    actual_path = path_pattern.replace('{timestamp}', timestamp)
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(actual_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Save the graph
    text_graph.save_graph(actual_path)
    
    return actual_path

def save_outputs(text_graph, args):
    """Save various output formats"""
    config = load_config()
    outputs_saved = []
    
    # Save graph file
    export_graph = getattr(args, 'export_graph', config['export_graph'])
    if export_graph:
        try:
            text_graph.save_graph(export_graph)
            outputs_saved.append(f"Graph: {export_graph}")
        except Exception as e:
            print(f"⚠️ Failed to save graph: {e}")
    
    # Save JSON data
    if getattr(args, 'export_json', None):
        try:
            json_data = text_graph.export_to_json()
            with open(args.export_json, 'w', encoding='utf-8') as f:
                f.write(json_data)
            outputs_saved.append(f"JSON: {args.export_json}")
        except Exception as e:
            print(f"⚠️ Failed to save JSON: {e}")
    
    # Handle visualization
    if not getattr(args, 'disable_visualization', False):
        try:
            # Parse figure size
            if hasattr(args, 'figure_size'):
                fig_width, fig_height = map(float, args.figure_size.split(','))
            else:
                fig_width, fig_height = config['figure_width'], config['figure_height']
            
            # Create visualization
            text_graph.visualize(
                figsize=(fig_width, fig_height),
                show_dependencies=not getattr(args, 'disable_dependencies', False),
                show_semantic=not getattr(args, 'disable_semantic', False)
            )
            
            # Save image if specified
            if getattr(args, 'export_image', None):
                plt.savefig(
                    args.export_image, 
                    dpi=config['dpi'], 
                    bbox_inches='tight', 
                    facecolor='white'
                )
                outputs_saved.append(f"Image: {args.export_image}")
                
                # Don't show plot if saving to file
                plt.close()
            else:
                # Show plot
                plt.show()
                
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
    
    # Print saved outputs
    if outputs_saved and not getattr(args, 'quiet', False):
        print(f"💾 Saved outputs:")
        for output in outputs_saved:
            print(f"  ✅ {output}")

def process_multiple_samples(args):
    """Xử lý tất cả samples từ file input với beam search"""
    import datetime
    
    if not args.input_file:
        raise ValueError("Cần có input file để xử lý multiple samples")
    
    if not os.path.exists(args.input_file):
        raise ValueError(f"File không tồn tại: {args.input_file}")
    
    # Đọc file JSON
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"File JSON không hợp lệ: {args.input_file}")
    
    if not isinstance(data, list):
        raise ValueError("File phải chứa array của samples")
    
    if len(data) == 0:
        raise ValueError("File không chứa samples nào")
    
    print(f"🚀 Bắt đầu xử lý {len(data)} samples với beam search...")
    
    # Setup VnCoreNLP một lần
    config = load_config()
    model = setup_vncorenlp(config['vncorenlp_path'], args.verbose)
    
    # Tạo thư mục output cho multiple samples
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = f"output_multiple_{timestamp}"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Auto-detect device và áp dụng optimizations
    device_info = detect_device()
    apply_device_optimizations(args, device_info, args.verbose)
    
    results = []
    failed_samples = []
    
    for idx, sample in enumerate(data):
        print(f"\n📋 Processing sample {idx+1}/{len(data)}...")
        
        try:
            # Validate sample format
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {idx+1} không phải dict")
            
            context = sample.get('context', '')
            claim = sample.get('claim', '')
            label = sample.get('label', None)
            evidence = sample.get('evidence', '')
            
            if not context or not claim:
                raise ValueError(f"Sample {idx+1} thiếu context hoặc claim")
            
            # Process this sample
            context_sentences, claim_sentences = process_text_data(
                model, context, claim, args.verbose
            )
            
            # Build graph cho sample này
            text_graph = build_complete_graph(
                context, claim, context_sentences, claim_sentences, args
            )
            
            # Save outputs cho sample này
            sample_output_dir = os.path.join(output_base_dir, f"sample_{idx+1:03d}")
            os.makedirs(sample_output_dir, exist_ok=True)
            
            # Save graph
            graph_file = os.path.join(sample_output_dir, f"graph_sample_{idx+1:03d}.gexf")
            text_graph.save_graph(graph_file)
            
            # Save JSON 
            json_file = os.path.join(sample_output_dir, f"output_sample_{idx+1:03d}.json")
            json_data = text_graph.export_to_json()
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(json_data)
            
            # Collect statistics
            sample_result = {
                'sample_id': idx + 1,
                'graph_file': graph_file,
                'json_file': json_file,
                'nodes': text_graph.graph.number_of_nodes(),
                'edges': text_graph.graph.number_of_edges(),
                'entities': len([n for n in text_graph.graph.nodes() if text_graph.graph.nodes[n].get('type') == 'entity']),
                'claim': claim[:100] + "..." if len(claim) > 100 else claim,
                'label': label,
                'beam_search_paths': 0  # Sẽ được update nếu có beam search
            }
            
            # Beam search paths nếu có
            if hasattr(text_graph, '_last_beam_search_results') and text_graph._last_beam_search_results:
                sample_result['beam_search_paths'] = len(text_graph._last_beam_search_results)
                sample_result['beam_search_files'] = getattr(text_graph, '_last_beam_search_files', [])
            
            results.append(sample_result)
            
            if args.verbose:
                print(f"  ✅ Sample {idx+1} processed: {sample_result['nodes']} nodes, {sample_result['edges']} edges")
        
        except Exception as e:
            failed_samples.append({'sample_id': idx + 1, 'error': str(e)})
            if args.verbose:
                print(f"  ❌ Sample {idx+1} failed: {e}")
    
    # Tạo summary report
    summary_file = os.path.join(output_base_dir, "processing_summary.json")
    summary = {
        'total_samples': len(data),
        'successful_samples': len(results),
        'failed_samples': len(failed_samples),
        'timestamp': timestamp,
        'results': results,
        'failures': failed_samples,
        'configuration': {
            'beam_search': getattr(args, 'beam_search', False),
            'beam_width': getattr(args, 'beam_width', 10),
            'beam_max_depth': getattr(args, 'beam_max_depth', 6),
            'similarity_threshold': getattr(args, 'similarity_threshold', 0.85),
            'top_k': getattr(args, 'top_k', 5)
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Print final summary
    print(f"\n🎉 HOÀN THÀNH XỬ LÝ MULTIPLE SAMPLES!")
    print(f"📊 Tổng quan:")
    print(f"  - Tổng samples: {len(data)}")
    print(f"  - Thành công: {len(results)}")
    print(f"  - Thất bại: {len(failed_samples)}")
    print(f"  - Output directory: {output_base_dir}")
    print(f"  - Summary file: {summary_file}")
    
    if failed_samples:
        print(f"\n❌ Failed samples:")
        for failure in failed_samples[:5]:  # Show first 5 failures
            print(f"  - Sample {failure['sample_id']}: {failure['error']}")
        if len(failed_samples) > 5:
            print(f"  ... và {len(failed_samples) - 5} samples khác")
    
    return results

def load_sample_data():
    """Load sample data for demo (deprecated, use load_demo_data instead)"""
    return load_demo_data()

def download_vncorenlp(target_dir="vncorenlp", verbose=False):
    """Download and setup VnCoreNLP automatically using py_vncorenlp"""
    
    # Convert to absolute path
    if not os.path.isabs(target_dir):
        target_dir = os.path.abspath(target_dir)
    
    # Check if already exists
    jar_path = os.path.join(target_dir, "VnCoreNLP-1.2.jar")
    models_path = os.path.join(target_dir, "models")
    
    if os.path.exists(jar_path) and os.path.exists(models_path):
        if verbose:
            print(f"✅ VnCoreNLP already exists at: {target_dir}")
        return target_dir
    
    if verbose:
        print(f"📥 Downloading VnCoreNLP to: {target_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Use py_vncorenlp's built-in download function
        import py_vncorenlp
        py_vncorenlp.download_model(save_dir=target_dir)
        
        if verbose:
            print("  ✅ VnCoreNLP downloaded successfully!")
        
        return target_dir
        
    except Exception as e:
        raise RuntimeError(f"Failed to download VnCoreNLP: {e}") 

def segment_entity_with_vncorenlp(entity, model):
    """
    Segment entity sử dụng VnCoreNLP để match với segmented text
    
    Args:
        entity (str): Entity text cần segment
        model: VnCoreNLP model instance
        
    Returns:
        str: Segmented entity với words nối bằng underscore
    """
    try:
        result = model.annotate_text(entity)
        segmented_words = []
        if result and len(result) > 0:
            first_sentence = list(result.values())[0]
            for token in first_sentence:
                segmented_words.append(token["wordForm"])
        
        segmented_entity = "_".join(segmented_words)
        return segmented_entity
        
    except Exception as e:
        # Fallback: simple space to underscore replacement
        return entity.replace(" ", "_") 