### **1. 聚类分析**

#### **背景**

代码中使用了 KMeans 聚类算法对食谱的营养特征进行分组（`nutrition_clusters`），以发现具有相似营养特性的食谱。这属于无监督学习的范畴。

#### **算法设计：基于 KMeans 的营养特征聚类**

**目标**：将食谱按其营养特性（如卡路里、蛋白质、脂肪等）划分为若干簇，便于后续推荐和分析。

**步骤**：

1. **数据准备**

   - 提取食谱的营养特征列（如 `Calories`, `FatContent`, `ProteinContent` 等）。
   - 对缺失值进行过滤，确保每个样本都有完整的营养特征数据。
   - 使用 `VectorAssembler` 将多维特征向量组合为一个特征列。

2. **特征归一化**

   - 由于不同营养特征的数值范围差异较大（例如，卡路里可能在几百到几千，而脂肪含量通常是个位数），需要对特征进行归一化处理。
   - 使用 `MinMaxScaler` 将特征缩放到 [0, 1] 范围。

3. **KMeans 聚类**

   - 设定聚类数量 $ k $（代码中为 10）。
   - 训练 KMeans 模型，将食谱分配到不同的簇。
   - 输出每个簇的中心点以及分配结果。

4. **结果分析**
   - 统计每个簇的食谱数量。
   - 分析每个簇的平均营养特征（如平均卡路里、平均蛋白质含量等），以解释簇的意义。
   - 可视化簇分布（如二维散点图，横轴为卡路里，纵轴为蛋白质含量，颜色表示簇标签）。

**伪代码**：

```python
# 1. 数据准备
nutrition_df = recipes_df.select("RecipeId", "Calories", "FatContent", ..., "SodiumContent")
filtered_nutrition = nutrition_df.dropna()

# 2. 特征归一化
assembler = VectorAssembler(inputCols=["Calories", "FatContent", ...], outputCol="features")
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
scaled_data = scaler.fit(assembler.transform(filtered_nutrition))

# 3. KMeans 聚类
kmeans = KMeans(k=10, seed=42, featuresCol="scaled_features")
clusters = kmeans.fit(scaled_data).transform(scaled_data)

# 4. 结果分析
cluster_analysis = clusters.groupBy("prediction").agg(
    count("*").alias("recipe_count"),
    avg("Calories").alias("avg_calories"),
    avg("ProteinContent").alias("avg_protein")
)
```

### **2. 食谱推荐系统**

#### **背景**

代码实现了一个混合推荐系统，结合了协同过滤（ALS 矩阵分解）和基于内容的推荐（营养、类别、成分）。以下是协同过滤部分的设计。

#### **算法设计：基于 ALS 的协同过滤推荐**

**目标**：根据用户的历史评分行为，预测用户对未评分食谱的兴趣，生成个性化推荐列表。

**步骤**：

1. **数据预处理**

   - 从用户-食谱交互数据中提取三元组 `(user_id, item_id, rating)`。
   - 对用户 ID 和食谱 ID 进行索引化（StringIndexer）。
   - 过滤无效评分（如为空或超出范围的评分）。

2. **矩阵分解模型训练**

   - 使用 ALS（交替最小二乘法）模型分解用户-食谱评分矩阵。
   - 模型参数包括：
     - `rank`：隐因子维度（代码中为 50）。
     - `maxIter`：最大迭代次数（代码中为 10）。
     - `regParam`：正则化参数（代码中为 0.1）。
   - 划分训练集和测试集（8:2）。

3. **预测与评估**

   - 在测试集上生成预测评分。
   - 使用 RMSE（均方根误差）评估模型性能。

4. **生成推荐**
   - 对目标用户生成 Top-N 推荐列表。
   - 使用 `recommendForUserSubset` 方法生成批量用户的推荐。

**伪代码**：

```python
# 1. 数据预处理
indexed_data = user_indexer.fit(user_item_df).transform(user_item_df)
(training, test) = indexed_data.randomSplit([0.8, 0.2], seed=42)

# 2. 矩阵分解模型训练
als = ALS(maxIter=10, regParam=0.1, rank=50, userCol="user_id", itemCol="item_id", ratingCol="Rating")
als_model = als.fit(training)

# 3. 预测与评估
predictions = als_model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="Rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

# 4. 生成推荐
sample_users = top_users.select("user_id").limit(5)
user_recs = als_model.recommendForUserSubset(sample_users, 10)
```
