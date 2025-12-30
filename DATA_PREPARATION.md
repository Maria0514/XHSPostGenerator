# 小红书文案知识库构建指南

## 📚 概述

本文档指导如何为小红书文案生成器构建高质量的RAG知识库。

---

## 🎯 目标

为每个内容类型构建一个包含50-100条优质小红书文案的知识库，用于RAG检索。

---

## 📁 文件结构

构建完成后的目录结构：

```
temp/
├── xiaohongshu_1-lifestyle/
│   ├── travel_tips.txt
│   ├── daily_life.txt
│   ├── food_sharing.txt
│   └── home_decoration.txt
├── xiaohongshu_2-beauty/
│   ├── skincare_routine.txt
│   ├── makeup_reviews.txt
│   ├── product_recommendations.txt
│   └── beauty_tips.txt
├── xiaohongshu_3-fashion/
│   ├── outfit_inspiration.txt
│   ├── style_guide.txt
│   ├── seasonal_trends.txt
│   └── matching_tips.txt
├── xiaohongshu_4-fitness/
│   ├── workout_plans.txt
│   ├── weight_loss_stories.txt
│   ├── fitness_tips.txt
│   └── nutrition_guide.txt
├── xiaohongshu_5-tech/
│   ├── product_reviews.txt
│   ├── tech_tips.txt
│   ├── gadget_recommendations.txt
│   └── photography_guide.txt
├── xiaohongshu_6-entertainment/
│   ├── music_recommendations.txt
│   ├── drama_reviews.txt
│   ├── movie_recommendations.txt
│   └── entertainment_news.txt
├── xiaohongshu_7-reading/
│   ├── book_reviews.txt
│   ├── reading_tips.txt
│   ├── author_introduction.txt
│   └── reading_recommendations.txt
└── xiaohongshu_8-pets/
    ├── pet_care_guide.txt
    ├── pet_stories.txt
    ├── training_tips.txt
    └── pet_health.txt
```

---

## 📝 文案收集方法

### 方法1：手动复制（推荐用于高质量数据）

1. 访问小红书官网 (https://www.xiaohongshu.com/)
2. 搜索相关关键词（如"咖啡馆探店"、"护肤心得"等）
3. 筛选高赞文案（点赞数 > 1000）
4. 复制文案内容到对应的txt文件

**优质文案的特征：**
- 点赞数 > 1000
- 评论数 > 100
- 内容原创且有价值
- 结构清晰，逻辑流畅
- 使用了emoji和话题标签

### 方法2：API爬取（需要技术能力）

使用 `tools/rag/` 中的工具进行自动爬取：

```bash
# 1. 使用url2article.md爬取网页内容
node tools/rag/url2article.md

# 2. 使用数据清洗工具过滤
python tools/rag/0-data_llm_filter.py

# 3. 转换为txt格式
python tools/rag/3-json2txt.py
```

### 方法3：使用现有数据集

从HuggingFace下载相关数据：
```bash
# 下载Tianji项目的RAG数据
huggingface-cli download sanbu/tianji-chinese --repo-type dataset
```

---

## ✍️ 文案整理规范

### 每个txt文件的格式

```
[标题1]
完整的小红书文案内容...
这是一篇完整的文案，包含开头、中间和结尾。
---

[标题2]
另一篇完整的小红书文案内容...
---

[标题3]
第三篇文案...
```

### 具体示例

**文件：** `temp/xiaohongshu_1-lifestyle/food_sharing.txt`

```
[咖啡馆探店：这家咖啡馆的手冲咖啡绝了]
😤 我终于明白了为什么别人的咖啡这么香！

昨天发现了一家隐藏在小巷里的咖啡馆，环境超舒服，老板还是个咖啡师。

✨ 环境：
- 复古装修，小资感十足
- 靠窗位置超适合拍照
- 背景音乐都是爵士乐

☕ 咖啡：
- 手冲咖啡超香，回甘明显
- 老板会根据豆子特性调整冲泡方式
- 价格也不贵，一杯才30块

🍰 配餐：
- 自制甜点，新鲜好吃
- 提拉米苏一定要试
- 可以免费加热

现在已经回头三次了，已经成为我的第二办公室😂

有没有其他隐藏咖啡馆推荐呀？留言告诉我~

#咖啡馆探店 #生活分享 #小资生活
---

[这个早餐组合让我瘦了5斤！]
🤔 你有没有想过，早餐吃什么竟然能影响一整天的状态？

前段时间开始调整早餐，没想到一个月竟然瘦了5斤！

我的早餐黄金组合：
🥑 牛油果 - 补充健康脂肪
🥚 水煮蛋 - 增加饱腹感
🍞 全麦面包 - 碳水均衡
🥗 新鲜果汁 - 补充维生素

为什么这个组合有效：
1. 高蛋白 - 早上吃蛋白质能提升代谢
2. 低糖 - 避免血糖波动
3. 高纤维 - 增加饱腹感，午餐吃得少

坚持了一个月，不仅瘦了，皮肤也变好了！

你们的早餐是怎样的？要不要一起来养成健康早餐的习惯？

#早餐分享 #减肥心得 #健康生活
---
```

---

## 🔍 数据质量检查清单

在添加文案到知识库前，检查以下项目：

- [ ] 文案长度 200-1000 字
- [ ] 内容原创，无明显抄袭
- [ ] 结构清晰：开头-中间-结尾
- [ ] 使用了 emoji（3-10个）
- [ ] 包含话题标签（#标签）
- [ ] 有互动元素（问题、投票等）
- [ ] 无违规内容
- [ ] 格式规范，易于阅读

---

## 📊 各内容类型的文案特征

### 1. 生活分享
- **特点**：日常、真实、有趣
- **常见元素**：故事、对比、建议
- **推荐话题**：#生活分享 #日常 #生活小技巧
- **示例数量**：50-80条

### 2. 美妆护肤
- **特点**：专业、详细、可操作
- **常见元素**：产品介绍、使用步骤、效果对比
- **推荐话题**：#护肤 #美妆 #产品推荐
- **示例数量**：60-100条

### 3. 时尚穿搭
- **特点**：视觉、搭配灵感、季节性
- **常见元素**：搭配方案、单品推荐、风格分析
- **推荐话题**：#穿搭 #时尚 #搭配灵感
- **示例数量**：50-80条

### 4. 运动健身
- **特点**：励志、专业、可执行
- **常见元素**：训练计划、进度分享、营养建议
- **推荐话题**：#健身 #运动 #减肥
- **示例数量**：50-80条

### 5. 科技数码
- **特点**：专业、详细、对比
- **常见元素**：产品评测、参数对比、使用体验
- **推荐话题**：#科技 #数码 #产品评测
- **示例数量**：40-60条

### 6. 音乐影视
- **特点**：情感、推荐、讨论
- **常见元素**：作品介绍、个人感受、剧情分析
- **推荐话题**：#音乐 #电视剧 #电影推荐
- **示例数量**：40-60条

### 7. 书籍阅读
- **特点**：思考、启发、推荐
- **常见元素**：书籍介绍、阅读感受、金句分享
- **推荐话题**：#阅读 #书籍推荐 #读书笔记
- **示例数量**：40-60条

### 8. 宠物生活
- **特点**：温暖、萌态、知识
- **常见元素**：宠物故事、护理知识、萌照分享
- **推荐话题**：#宠物 #萌宠 #养宠物
- **示例数量**：50-80条

---

## 🛠️ 自动化处理

### 使用LLM进行数据清洗

```bash
# 过滤低质量文案
python tools/rag/0-data_llm_filter.py

# 生成负样本（用于对比学习）
python tools/rag/0-data_llm_filter_negative.py

# 过滤过短文本
python tools/rag/0-data_llm_filter_lesswords.py
```

### 知识库优化

```bash
# 进行知识聚类分析
python tools/rag/2-jsonknowledges_kmeans.py

# 生成知识摘要
python tools/rag/1-get_rag_knowledges.py
```

---

## ⚡ 快速构建步骤

### 最小化方案（快速开始）

1. **收集基础文案** (2-3小时)
   - 每个内容类型收集 10-20 条优质文案
   - 放入对应的 txt 文件
   - 总共 80-160 条文案

2. **测试系统** (30分钟)
   ```bash
   streamlit run run/demo_agent_metagpt.py
   ```

3. **迭代优化** (持续)
   - 根据生成效果调整文案
   - 补充更多高质量文案

### 完整方案（高质量）

1. **系统收集** (1周)
   - 每个内容类型收集 50-100 条文案
   - 进行质量筛选
   - 总共 400-800 条文案

2. **数据清洗** (2-3天)
   - 使用LLM工具进行清洗
   - 去重和格式统一
   - 生成知识摘要

3. **知识库构建** (1天)
   - 构建向量数据库
   - 进行性能测试
   - 优化检索效果

4. **系统测试** (2-3天)
   - 全面功能测试
   - 文案质量评估
   - 用户体验优化

---

## 📈 数据收集渠道

### 官方渠道
- 小红书官网：https://www.xiaohongshu.com/
- 小红书创作者平台
- 小红书数据中心

### 第三方渠道
- 爬虫工具（需遵守robots.txt）
- 数据交易平台
- 公开数据集

### 社区资源
- GitHub上的开源数据集
- HuggingFace数据集
- Kaggle竞赛数据

---

## ✅ 验证清单

构建完成后，检查：

- [ ] 每个内容类型都有对应的文件夹
- [ ] 每个文件夹包含至少10个txt文件
- [ ] 每个txt文件包含3-5篇文案
- [ ] 所有文案格式统一
- [ ] 没有重复的文案
- [ ] 文案质量都在高水平
- [ ] 系统可以正常读取所有文件
- [ ] RAG检索能返回相关结果

---

## 🔗 相关资源

- [Tianji RAG文档](docs/RAG/rag_tutorial.md)
- [数据处理工具](tools/rag/)
- [LangChain文档](https://python.langchain.com/)
- [Chroma向量数据库](https://www.trychroma.com/)

---

## 💡 最佳实践

1. **优先质量而非数量**
   - 100条高质量文案 > 1000条低质量文案
   - 定期审查和更新数据

2. **保持多样性**
   - 不同风格的文案
   - 不同长度的文案
   - 不同角度的内容

3. **定期维护**
   - 定期更新热点内容
   - 移除过时的文案
   - 补充新的优质文案

4. **监控效果**
   - 记录生成文案的质量
   - 收集用户反馈
   - 根据反馈调整知识库

---

**准备好开始构建你的知识库了吗？** 🚀