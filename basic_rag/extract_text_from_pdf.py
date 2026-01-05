import fitz


def extract_text_from_pdf(pdf_path) -> str:
    """
    从 PDF 文件中提取文本内容
    :param pdf_path: 文件路径
    :return str: pdf文件全部文本内容
    """
    mypdf = fitz.open(pdf_path)
    all_text = ""

    # 遍历pdf的每一页
    for page in range(mypdf.page_count):
        my_page = mypdf[page]
        text = my_page.get_text("text")
        all_text += text

    return all_text

# 使用实例

pdf_path = "data/AI_Information.pdf"
extract_text = extract_text_from_pdf(pdf_path)
print(f"提取到的文本长度为: {len(extract_text)} 字符")