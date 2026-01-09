import re

def split_text_into_sentences(text):
    """将文本分割成句子"""
    # 简单的句子分割，使用句号、问号、感叹号
    # 先按句号分割，然后清理
    sentences = re.split(r'。', text.strip())
    sentences = [s.strip() + '。' for s in sentences if s.strip()]
    # 移除最后一个空句子
    if sentences and sentences[-1] == '。':
        sentences.pop()
    return sentences

def create_sliding_window_nodes(sentences, window_size=3):
    """创建滑动窗口节点"""
    nodes = []
    for i, sentence in enumerate(sentences):
        # 计算窗口范围
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)

        # 获取窗口内的句子
        window_sentences = sentences[start:end]
        window_text = ' '.join(window_sentences)

        # 创建节点
        node = {
            'id': i,
            'original_sentence': sentence,
            'window_text': window_text,
            'window_start': start,
            'window_end': end - 1
        }
        nodes.append(node)
    return nodes

# 示例文本
sample_text = """
《灭神纪∙猢狲》是一款动作角色扮演游戏。游戏背景设定在架空的神话世界中。
玩家将扮演齐天大圣孙悟空，在充满东方神话元素的世界中展开冒险。
游戏的战斗系统极具特色，采用了独特的"变身系统"。悟空可以在战斗中变换不同形态。
每种形态都有其独特的战斗风格和技能组合。金刚形态侧重力量型打击，带来压倒性的破坏力。
魔佛形态则专注法术攻击，能释放强大的法术伤害。
游戏世界中充满了标志性的神话角色，除了主角孙悟空以外，还有来自佛教、道教等各派系的神魔。
这些角色既可能是悟空的盟友，也可能是需要击败的强大对手。
装备系统包含了丰富的武器选择，除了著名的如意金箍棒以外，悟空还可以使用各种神器法宝。
不同武器有其特色效果，玩家需要根据战斗场景灵活选择。
游戏的画面表现极具东方美学特色，场景融合了水墨画风格，将山川、建筑等元素完美呈现。
战斗特效既有中国传统文化元素，又具备现代游戏的视觉震撼力。
难度设计上，Boss战充满挑战性，需要玩家精准把握战斗节奏和技能运用。
同时游戏也提供了多种难度选择，照顾不同技术水平的玩家。
"""

print("=== 节点句子滑动窗口示例 ===\n")

# 分割句子
sentences = split_text_into_sentences(sample_text)
print(f"文本被分割成 {len(sentences)} 个句子：\n")
for i, sentence in enumerate(sentences):
    print(f"{i}: {sentence}")
print("\n" + "="*50 + "\n")

# 创建滑动窗口节点
window_size = 3
nodes = create_sliding_window_nodes(sentences, window_size)

print(f"使用窗口大小 {window_size} 创建节点：\n")
for node in nodes:
    print(f"节点 {node['id']}:")
    print(f"  原始句子: {node['original_sentence']}")
    print(f"  窗口范围: 句子 {node['window_start']} 到 {node['window_end']}")
    print(f"  窗口内容: {node['window_text']}")
    print("-" * 50)

print("\n=== 索引优化效果演示 ===")
print("通过滑动窗口，每个节点不仅包含原始句子，还包含周围的上下文信息。")
print("这有助于在检索时提供更丰富的上下文，提升问答质量。")
print("例如，在回答关于'悟空形态变化'的问题时，窗口可以提供相关的上下文句子。")