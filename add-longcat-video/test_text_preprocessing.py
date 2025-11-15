#!/usr/bin/env python3
"""
验证 FastVideo 和 LongCat 的文本预处理是否一致
"""

import sys
import html
import ftfy
import regex as re


def longcat_original_preprocess(text: str) -> str:
    """原始 LongCat 的文本预处理函数"""
    # basic_clean
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = text.strip()
    # whitespace_clean
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def test_text_preprocessing():
    """测试文本预处理"""
    
    # 从 FastVideo 导入配置
    from fastvideo.configs.pipelines.longcat import longcat_preprocess_text
    
    # 测试用例
    test_cases = [
        # 正常文本
        "In a realistic photography style, an asian boy sits on a park bench.",
        
        # 包含多余空格
        "Text  with   multiple    spaces",
        
        # 包含 HTML 实体
        "Text with &amp; ampersand &lt;tag&gt;",
        
        # 包含 Unicode 问题
        'Text with "smart quotes" and —dashes—',
        
        # 换行符和制表符
        "Text\nwith\twhitespace\n\ncharacters",
        
        # 实际的 prompt
        "In a realistic photography style, an asian boy around seven or eight years old sits on a park bench, wearing a light yellow T-shirt, denim shorts, and white sneakers.",
    ]
    
    print("=" * 80)
    print("文本预处理一致性测试")
    print("=" * 80)
    
    all_passed = True
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}:")
        print(f"   原始: {repr(test_text)}")
        
        # 使用原始 LongCat 函数
        original_result = longcat_original_preprocess(test_text)
        
        # 使用 FastVideo 的函数
        fastvideo_result = longcat_preprocess_text(test_text)
        
        # 比较结果
        if original_result == fastvideo_result:
            print(f"   ✅ 一致: {repr(original_result)}")
        else:
            print(f"   ❌ 不一致!")
            print(f"      原始 LongCat: {repr(original_result)}")
            print(f"      FastVideo:     {repr(fastvideo_result)}")
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有测试通过！FastVideo 和 LongCat 的文本预处理完全一致。")
        return 0
    else:
        print("❌ 部分测试失败！请检查实现差异。")
        return 1


if __name__ == "__main__":
    sys.exit(test_text_preprocessing())

