#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test độ tương đồng giữa 2 câu cụ thể để đề xuất ngưỡng phù hợp
"""

import sys
import os
sys.path.append('.')

from mint.text_graph import TextGraph

def test_sentence_similarity():
    """Test độ tương đồng giữa 2 câu cụ thể"""
    
    # Khởi tạo TextGraph
    text_graph = TextGraph()
    
    # 2 câu cần so sánh - CẶP THỨ 2
    sentence1 = "Thay vì cố gắng dạy chim nói tiếng Anh, các nhà nghiên cứu đang giải mã những gì chúng nói với nhau bằng tiếng chim bằng cách tận dụng công nghệ mới để hiểu được giao tiếp của động vật."
    
    sentence2 = "Các biên tập viên trang Scientific American đã có cuộc trò chuyện với giáo sư Karen Bakker tại Đại học British Columbia ( Canada ) và là thành viên tại Viện Nghiên cứu cao cấp Harvard Radcliffe ( Mỹ ) về cách các nhà nghiên cứu đang tận dụng công nghệ mới để hiểu được giao tiếp của động vật"
    
    print("🔍 TEST ĐỘ TƯƠNG ĐỒNG GIỮA 2 CÂU")
    print("=" * 60)
    print(f"📝 Câu 1: {sentence1}")
    print(f"📝 Câu 2: {sentence2}")
    print()
    
    # Kiểm tra xem PhoBERT có khả dụng không
    if text_graph.phobert_model is None or text_graph.phobert_tokenizer is None:
        print("❌ PhoBERT model chưa được khởi tạo. Không thể tính toán.")
        return
    
    # Tính độ tương đồng
    print("🧠 Đang tính toán độ tương đồng với PhoBERT...")
    similarity = text_graph.get_sentence_similarity(sentence1, sentence2)
    
    print(f"🎯 KẾT QUỀ: {similarity:.4f}")
    print()
    
    # Phân tích kết quả
    print("📊 PHÂN TÍCH:")
    print(f"   Độ tương đồng: {similarity:.4f} ({similarity*100:.2f}%)")
    
    if similarity >= 0.90:
        print("   🔥 Rất cao - hai câu gần như giống nhau về mặt ngữ nghĩa")
        recommended_threshold = 0.85
    elif similarity >= 0.80:
        print("   ✅ Cao - hai câu có ý nghĩa tương đồng mạnh")
        recommended_threshold = 0.75
    elif similarity >= 0.70:
        print("   ⚠️ Trung bình - hai câu có một số điểm chung")
        recommended_threshold = 0.65
    elif similarity >= 0.60:
        print("   📉 Thấp - hai câu có ít điểm chung")
        recommended_threshold = 0.55
    else:
        print("   ❌ Rất thấp - hai câu khác nhau về mặt ngữ nghĩa")
        recommended_threshold = 0.50
    
    print()
    print("💡 ĐỀ XUẤT NGƯỠNG:")
    print(f"   Dựa trên kết quả này, tôi đề xuất ngưỡng: {recommended_threshold:.2f}")
    print()
    
    # Test các ngưỡng khác nhau
    thresholds_to_test = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    print("🎛️ SO SÁNH VỚI CÁC NGƯỠNG KHÁC NHAU:")
    for threshold in thresholds_to_test:
        if similarity >= threshold:
            status = "✅ SẼ KẾT NỐI"
        else:
            status = "❌ KHÔNG KẾT NỐI"
        print(f"   Ngưỡng {threshold:.2f}: {status}")
    
    print()
    print("🏆 KẾT LUẬN:")
    print(f"   - Với cặp câu này (similarity = {similarity:.3f})")
    print(f"   - Ngưỡng phù hợp nhất: {recommended_threshold:.2f}")
    print(f"   - Ngưỡng này sẽ bắt được các câu tương đồng mạnh mà không quá lỏng lẻo")
    
    return similarity, recommended_threshold

def suggest_threshold_strategy():
    """Đề xuất chiến lược chọn ngưỡng"""
    print("\n🎯 CHIẾN LƯỢC CHỌN NGƯỠNG CHO FACT-CHECKING:")
    print("=" * 50)
    
    strategies = [
        (0.85, "Nghiêm ngặt", "Chỉ kết nối các câu rất tương đồng. Ít false positive nhưng có thể bỏ sót."),
        (0.80, "Cân bằng cao", "Kết nối các câu có ý nghĩa tương đồng mạnh. Phù hợp cho fact-checking chính xác."),
        (0.75, "Cân bằng", "Kết nối đa số câu có liên quan. Khuyến nghị cho mục đích tổng quát."),
        (0.70, "Rộng rãi", "Kết nối nhiều câu, có thể có một số noise. Phù hợp khi muốn tìm mọi liên quan."),
        (0.65, "Rất rộng", "Kết nối rất nhiều câu. Có thể có nhiều false positive.")
    ]
    
    for threshold, name, description in strategies:
        print(f"\n🎚️ {threshold:.2f} - {name}:")
        print(f"   {description}")
    
    print(f"\n💡 Dựa trên test thực tế, tôi khuyến nghị bắt đầu với 0.75-0.80")

if __name__ == "__main__":
    similarity, recommended = test_sentence_similarity()
    suggest_threshold_strategy()
    
    print(f"\n✨ Để áp dụng ngưỡng đề xuất ({recommended:.2f}):")
    print(f"text_graph.build_sentence_claim_semantic_edges(similarity_threshold={recommended:.2f})") 