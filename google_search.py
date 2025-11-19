import json
import re
from ddgs import DDGS


def smart_search(keyword, target_count=5):
    """
    智能搜索函数 V2：
    1. 支持正则模糊匹配 (解决 Gemini3 vs Gemini 3.0)
    2. 支持上下文语义验证 (剔除 USB 线材、食品标准等无关内容)
    3. 优先搜索 News，降级使用 Text
    """

    print(f"🔍 开始搜索: '{keyword}' (目标: {target_count} 条)")

    # --- 核心升级：更智能的验证逻辑 ---
    def is_valid_result(item, original_query):
        # 1. 准备数据
        title = item.get('title', '').lower()
        body = (item.get('body') or item.get('snippet') or '').lower()
        content = title + " " + body
        query_lower = original_query.lower()

        # --- A. 针对 Gemini3 的特殊正则处理 ---
        if "gemini" in query_lower and "3" in query_lower:
            # 正则解释：匹配 gemini，后面接任意个(空格/横杠/点)，然后接3
            # 能匹配: gemini3, gemini 3, gemini-3, gemini 3.0
            if not re.search(r'gemini[\s\-\.]*3', content):
                return False

            # 【上下文锚点】必须包含以下词汇之一，防止匹配到 "iFi Audio Gemini3.0" 线材
            context_anchors = ['google', 'ai', 'llm', 'model', 'intelligence', 'reasoning']
            if not any(anchor in content for anchor in context_anchors):
                return False

            # 如果原词里有 pro，那么结果里也必须有 pro
            if "pro" in query_lower and "pro" not in content:
                return False

            return True

        # --- B. 针对 Codex 的特殊处理 ---
        if "codex" in query_lower:
            # 必须包含 codex
            if "codex" not in content:
                return False
            # 如果搜 OpenAI，必须包含 OpenAI
            if "openai" in query_lower and "openai" not in content:
                return False

            # 【负面词过滤】排除食品、拼图等干扰
            forbidden = ['cashew', 'food', 'puzzle', 'game', 'nintendo', 'silenda']
            if any(bad in content for bad in forbidden):
                return False

            return True

        # --- C. 默认逻辑：词根全包含 ---
        # 对于 "nano banana" 这种，继续使用切分匹配
        required_terms = query_lower.split()
        return all(term in content for term in required_terms)

    # --- 标准化函数 ---
    def normalize_item(item, source_type):
        url = item.get('url') or item.get('href')
        return {
            "title": item.get('title'),
            "body": item.get('body') or item.get('snippet'),
            "url": url,
            "date": item.get('date') or "Unknown",
            "source": item.get('source') or "Web Search",
            "type": source_type
        }

    final_results = []
    seen_urls = set()

    # --- 优化查询词 ---
    # 搜索引擎通常对分开的词理解更好，比如搜 "Gemini 3" 比 "Gemini3" 结果多
    search_query = keyword.replace("Gemini3", "Gemini 3")

    # ==========================================
    # 阶段 1: News 搜索
    # ==========================================
    print("1️⃣  正在进行 News 搜索...")
    try:
        news_gen = DDGS().news(
            query=f'"{search_query}"',  # 使用优化后的关键词
            region="us-en",
            safesearch="off",
            timelimit="m",
            max_results=target_count * 3
        )

        for res in news_gen:
            if len(final_results) >= target_count: break

            url = res.get('url')
            # 传入原始 keyword 用于判断逻辑
            if url not in seen_urls and is_valid_result(res, keyword):
                final_results.append(normalize_item(res, "News"))
                seen_urls.add(url)

    except Exception as e:
        print(f"   ⚠️ News 搜索出现问题: {e}")

    print(f"   -> News 阶段获取到 {len(final_results)} 条有效结果。")

    # ==========================================
    # 阶段 2: Text 搜索
    # ==========================================
    needed = target_count - len(final_results)

    if needed > 0:
        print(f"2️⃣  数量不足，补齐 {needed} 条 (Text 搜索)...")
        try:
            text_gen = DDGS().text(
                query=f'"{search_query}"',  # 使用优化后的关键词
                region="us-en",
                safesearch="off",
                timelimit="y",
                max_results=needed * 5,
                # backends="google" # 注：新版库通常不需要指定 backend，去掉以防报错
            )

            for res in text_gen:
                if len(final_results) >= target_count: break

                url = res.get('href')
                if url not in seen_urls and is_valid_result(res, keyword):
                    final_results.append(normalize_item(res, "Web Text"))
                    seen_urls.add(url)

        except Exception as e:
            print(f"   ⚠️ Text 搜索出现问题: {e}")
    else:
        print("✅ News 结果已满足数量，跳过 Text 搜索。")

    return final_results


# ==========================================
# 运行测试
# ==========================================
if __name__ == "__main__":
    # 测试列表：包含容易歧义的词和写法不规范的词
    list_keywords = ["OpenAI Codex", "Gemini3 Pro", "Antigravity", "nano banana"]

    for query_keyword in list_keywords:
        count_needed = 10  # 设为 5 条方便测试
        results = smart_search(query_keyword, count_needed)

        print("\n" + "=" * 40)
        print(f"最终结果: {query_keyword} (共 {len(results)} 条)")
        print("=" * 40)
        # 只打印前2条的预览，防止控制台刷屏太长
        if results:
            print(json.dumps(results, indent=4, ensure_ascii=False))
            # print(f"Title [0]: {results[0]['title']}")
            # print(f"Type  [0]: {results[0]['type']}")
            # print(f"URL   [0]: {results[0]['url']}")
        else:
            print("❌ 未找到结果")