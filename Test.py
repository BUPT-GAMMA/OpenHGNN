def cyclic_shift(words):
    # 将单词列表进行循环移位
    shifted_words = []
    for i in range(len(words)):
        shifted = words[i:] + words[:i]  # 进行循环移位
        shifted_words.append(shifted)  # 添加移位后的单词列表到列表中
    return shifted_words

def process_text(text):
    lines = text.split('\n')  # 按行分割文本
    for line in lines:
        line = line.strip().upper()  # 去除首尾空格并转换为大写字母
        if line:  # 如果该行不为空
            words = line.split()  # 按空格分割单词
            shifted_sentences = cyclic_shift(words)  # 对单词列表进行循环移位
            for sentence in shifted_sentences:  # 输出每个循环移位后的句子
                print(' '.join(sentence))  # 将单词列表拼接成句子并输出





if __name__ == "__main__":
# 测试输入
    text = """HELLO WORLD
    PYTHON IS GREAT
    """
    process_text(text)