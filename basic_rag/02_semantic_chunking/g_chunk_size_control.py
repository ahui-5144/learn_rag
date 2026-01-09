
def create_controlled_chunks(sentences, similarities,
                             max_chunk_size=1000, min_chunk_size=200):
    """
    控制分块大小的语义分块
    Args:
        sentences: 句子列表
        similarities: 相邻句子相似度列表
        max_chunk_size: 单个分块最大字符数（超过强制切分）
        min_chunk_size: 单个分块最小字符数（低于此值不按语义切分）

    Returns:

    """
    chunks = []
    current_chunk = []
    current_size  = 0

    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        current_size  += len(sentence)

        # 检查是否需要分块
        should_break = False

        if i < len(similarities):
            # 如果相似度低且(不需要合并)当前块大小合适
            if (similarities[i] < 0.5 and
                current_size >= min_chunk_size):
                should_break = True

        # 如果块太大，强制分块
        if current_size >= max_chunk_size:
            should_break = True

        if should_break or i == len(sentences) - 1:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            current_chunk = []
            current_size = 0

    return chunks
