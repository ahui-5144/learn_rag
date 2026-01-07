
def split_into_sentences(text):
    """
    将文本分割为句子列表
    Args:
        text:

    Returns:

    """
    sentences = text.split("。")
    sentences = [s.strip() + "." for s in sentences if s.strip()]
    return sentences

# 示例
text = "人工智能是计算机科学的分支。它研究智能的本质。AI可以模拟人类思维。"
sentences = split_into_sentences(text)
print(f"句子列表：{sentences}" )