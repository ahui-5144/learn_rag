from extract_text_from_pdf import extract_text_from_pdf


def chunk_text(text, n, overlap) -> list[str]:
    """
    将文本分割成固定大小的块，支持重叠

    Args:
        text: 原始文本
        n: 每块的字符数
        overlap: 重叠的字符数

    Returns: 分隔好的文本块list

    """
    chunks = []

    # 步长 = 块大小 - 重叠大小
    step = n - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + n] # Python 字符串切片语法 text[start:end]：取从 start（包含）到 end（不包含）的子串
        chunks.append(chunk)

    return chunks


pdf_path = "data/AI_Information.pdf"
extract_text = extract_text_from_pdf(pdf_path)
# 使用示例
text_chunks = chunk_text(extract_text, 1000, 200)
print(f"创建了 {len(text_chunks)} 个文本块")
print(f"第一个文本块:\n{text_chunks[0]}")