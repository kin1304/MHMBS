#!/usr/bin/env python3
"""
📊 STATISTICS ANALYZER
Phân tích comprehensive statistics từ detailed output files
"""

import json
import numpy as np
from typing import Dict, List

def analyze_coverage_statistics(detailed_file_path: str):
    """
    Phân tích coverage statistics từ detailed output file
    """
    print("📊 VIETNAMESE MULTI-HOP FACT-CHECKING - STATISTICS ANALYSIS")
    print("=" * 70)
    
    # Load detailed results
    with open(detailed_file_path, 'r', encoding='utf-8') as f:
        detailed_results = json.load(f)
    
    total_samples = len(detailed_results)
    print(f"🔢 Total Samples Processed: {total_samples}")
    print()
    
    # Coverage Statistics Analysis
    print("📈 COVERAGE STATISTICS ANALYSIS")
    print("-" * 50)
    
    total_context_sentences_all = []
    total_retrieved_sentences_all = []
    coverage_percentages = []
    
    for sample in detailed_results:
        stats = sample.get('statistics', {})
        coverage_stats = stats.get('coverage_statistics', {})
        
        total_context = coverage_stats.get('total_context_sentences', 0)
        total_retrieved = coverage_stats.get('total_retrieved_sentences', 0)
        coverage_pct = coverage_stats.get('coverage_percentage', 0)
        
        total_context_sentences_all.append(total_context)
        total_retrieved_sentences_all.append(total_retrieved)
        coverage_percentages.append(coverage_pct)
    
    # Overall Statistics
    total_context_sum = sum(total_context_sentences_all)
    total_retrieved_sum = sum(total_retrieved_sentences_all)
    overall_coverage = (total_retrieved_sum / total_context_sum * 100) if total_context_sum > 0 else 0
    
    print(f"📄 Total Context Sentences (All Samples): {total_context_sum:,}")
    print(f"🎯 Total Retrieved Evidence Sentences: {total_retrieved_sum:,}")
    print(f"📊 Overall Coverage Percentage: {overall_coverage:.2f}%")
    print()
    
    # Per-Sample Statistics
    print("📋 PER-SAMPLE STATISTICS")
    print("-" * 50)
    print(f"📄 Context Sentences per Sample:")
    print(f"   - Average: {np.mean(total_context_sentences_all):.1f}")
    print(f"   - Median: {np.median(total_context_sentences_all):.1f}")
    print(f"   - Min: {min(total_context_sentences_all)}")
    print(f"   - Max: {max(total_context_sentences_all)}")
    print()
    
    print(f"🎯 Retrieved Evidence per Sample:")
    print(f"   - Average: {np.mean(total_retrieved_sentences_all):.1f}")
    print(f"   - Median: {np.median(total_retrieved_sentences_all):.1f}")
    print(f"   - Min: {min(total_retrieved_sentences_all)}")
    print(f"   - Max: {max(total_retrieved_sentences_all)}")
    print()
    
    print(f"📊 Coverage Percentage per Sample:")
    print(f"   - Average: {np.mean(coverage_percentages):.2f}%")
    print(f"   - Median: {np.median(coverage_percentages):.2f}%")
    print(f"   - Min: {min(coverage_percentages):.2f}%")
    print(f"   - Max: {max(coverage_percentages):.2f}%")
    print()
    
    # Multi-Hop Statistics Analysis
    print("🔍 MULTI-HOP REASONING STATISTICS")
    print("-" * 50)
    
    direct_sentences_counts = []
    multi_hop_sentences_counts = []
    multi_hop_levels_counts = []
    
    for sample in detailed_results:
        stats = sample.get('statistics', {})
        comprehensive_stats = stats.get('comprehensive_statistics', {})
        
        direct_count = comprehensive_stats.get('direct_sentences_count', 0)
        multi_hop_count = comprehensive_stats.get('total_multi_hop_sentences', 0)
        levels_count = comprehensive_stats.get('multi_hop_levels', 0)
        
        direct_sentences_counts.append(direct_count)
        multi_hop_sentences_counts.append(multi_hop_count)
        multi_hop_levels_counts.append(levels_count)
    
    print(f"🎯 Direct Connection Sentences:")
    print(f"   - Total: {sum(direct_sentences_counts)}")
    print(f"   - Average per sample: {np.mean(direct_sentences_counts):.1f}")
    print(f"   - Samples with direct connections: {sum(1 for x in direct_sentences_counts if x > 0)}/{total_samples}")
    print()
    
    print(f"🔍 Multi-Hop Sentences:")
    print(f"   - Total: {sum(multi_hop_sentences_counts)}")
    print(f"   - Average per sample: {np.mean(multi_hop_sentences_counts):.1f}")
    print(f"   - Max levels used: {max(multi_hop_levels_counts) if multi_hop_levels_counts else 0}")
    print()
    
    # Coverage Distribution Analysis
    print("📊 COVERAGE DISTRIBUTION ANALYSIS")
    print("-" * 50)
    
    coverage_ranges = {
        "Very Low (0-20%)": 0,
        "Low (21-40%)": 0,
        "Medium (41-60%)": 0,
        "High (61-80%)": 0,
        "Very High (81-100%)": 0
    }
    
    for coverage in coverage_percentages:
        if coverage <= 20:
            coverage_ranges["Very Low (0-20%)"] += 1
        elif coverage <= 40:
            coverage_ranges["Low (21-40%)"] += 1
        elif coverage <= 60:
            coverage_ranges["Medium (41-60%)"] += 1
        elif coverage <= 80:
            coverage_ranges["High (61-80%)"] += 1
        else:
            coverage_ranges["Very High (81-100%)"] += 1
    
    for range_name, count in coverage_ranges.items():
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"   {range_name}: {count} samples ({percentage:.1f}%)")
    print()
    
    # Efficiency Analysis
    print("⚡ EFFICIENCY ANALYSIS")
    print("-" * 50)
    
    # Samples với different evidence counts
    evidence_counts = {}
    for sample in detailed_results:
        evidence_count = len(sample.get('multi_level_evidence', []))
        evidence_counts[evidence_count] = evidence_counts.get(evidence_count, 0) + 1
    
    print(f"📋 Evidence Distribution:")
    for count in sorted(evidence_counts.keys()):
        sample_count = evidence_counts[count]
        percentage = (sample_count / total_samples * 100) if total_samples > 0 else 0
        print(f"   {count} evidence sentences: {sample_count} samples ({percentage:.1f}%)")
    print()
    
    # Success Rate Analysis
    successful_samples = sum(1 for sample in detailed_results if len(sample.get('multi_level_evidence', [])) > 0)
    success_rate = (successful_samples / total_samples * 100) if total_samples > 0 else 0
    
    print("🎯 SUCCESS RATE ANALYSIS")
    print("-" * 50)
    print(f"✅ Successful samples (with evidence): {successful_samples}/{total_samples} ({success_rate:.1f}%)")
    print(f"❌ Failed samples (no evidence): {total_samples - successful_samples}/{total_samples} ({100 - success_rate:.1f}%)")
    print()
    
    # Configuration Summary
    if detailed_results:
        config = detailed_results[0].get('statistics', {}).get('config', {})
        print("⚙️ CONFIGURATION USED")
        print("-" * 50)
        print(f"🔧 max_levels: {config.get('max_levels', 'N/A')}")
        print(f"🔧 beam_width_per_level: {config.get('beam_width_per_level', 'N/A')}")
        print(f"🔧 max_depth: {config.get('max_depth', 'N/A')}")
        print(f"🔧 max_final_sentences: {config.get('max_final_sentences', 'N/A')}")
        print(f"🔧 num_hops: {config.get('num_hops', 'N/A')}")
        print(f"🔧 filtering_approach: {config.get('filtering_approach', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("🎉 ANALYSIS COMPLETE!")

if __name__ == "__main__":
    # Auto-detect latest detailed file
    import os
    import glob
    
    pattern = "multi_hop_output/multi_hop_detailed_*.json"
    files = glob.glob(pattern)
    
    if files:
        latest_file = max(files, key=os.path.getctime)
        print(f"📁 Analyzing: {latest_file}")
        print()
        analyze_coverage_statistics(latest_file)
    else:
        print("❌ No detailed output files found!")
        print(f"   Looking for pattern: {pattern}") 