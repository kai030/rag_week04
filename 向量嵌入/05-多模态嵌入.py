"""
多模态嵌入简单示例：使用CLIP模型对图片进行编码
由于visual_bge不可用，使用CLIP作为替代
"""

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import numpy as np
    import requests
    from io import BytesIO

    # 初始化编码器
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 定义图片路径（使用网络图片作为示例）
    image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # 对图片进行编码
    with torch.no_grad():
        # 仅使用图片进行编码
        inputs = processor(images=image, return_tensors="pt")
        image_features = model.get_image_features(**inputs)
        image_embedding = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 使用图片和文本进行编码
        text = "这是一张悟空战斗示例图片"
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
        multimodal_features = model(**inputs)
        multimodal_embedding = multimodal_features.image_embeds / multimodal_features.image_embeds.norm(dim=-1, keepdim=True)

    # 将张量转移到CPU并转换为numpy数组
    image_embedding_np = image_embedding.cpu().numpy()
    multimodal_embedding_np = multimodal_embedding.cpu().numpy()

    # 打印嵌入向量的信息
    print("=== 图片嵌入向量信息 ===")
    print(f"向量维度: {image_embedding_np.shape[1]}")
    print(f"向量示例 (前10个元素): {image_embedding_np[0][:10]}")
    print(f"向量范数: {np.linalg.norm(image_embedding_np[0])}")

    print("\n=== 多模态嵌入向量信息 ===")
    print(f"向量维度: {multimodal_embedding_np.shape[1]}")
    print(f"向量示例 (前10个元素): {multimodal_embedding_np[0][:10]}")
    print(f"向量范数: {np.linalg.norm(multimodal_embedding_np[0])}")

except Exception as e:
    print(f"无法加载CLIP模型: {e}")
    print("使用模拟数据演示多模态嵌入。")
    import numpy as np
    # 模拟嵌入向量
    image_embedding_np = np.random.rand(1, 512)
    multimodal_embedding_np = np.random.rand(1, 512)
    
    print("=== 图片嵌入向量信息 (模拟) ===")
    print(f"向量维度: {image_embedding_np.shape[1]}")
    print(f"向量示例 (前10个元素): {image_embedding_np[0][:10]}")
    print(f"向量范数: {np.linalg.norm(image_embedding_np[0])}")

    print("\n=== 多模态嵌入向量信息 (模拟) ===")
    print(f"向量维度: {multimodal_embedding_np.shape[1]}")
    print(f"向量示例 (前10个元素): {multimodal_embedding_np[0][:10]}")
    print(f"向量范数: {np.linalg.norm(multimodal_embedding_np[0])}")

