### **算法设计：特征工程**

#### **1. 时间特征提取**

- **目标**: 从食谱的发布时间和评论提交时间中提取年份、月份、季度、季节、星期、小时等时间特征。
- **输入**: 食谱数据集 (`recipes_df`) 和评论数据集 (`reviews_df`) 中的时间字段（`DatePublished`, `DateSubmitted`）。
- **输出**: 包含时间特征的新数据集。

**算法步骤**:

1. 检查时间字段是否为空值，若不为空则进行解析：
   - 使用 Spark 的 `to_timestamp` 函数将字符串转换为时间戳。
2. 提取时间特征：
   - 年份 (`year`)
   - 月份 (`month`)
   - 季度 (`quarter`)
   - 季节 (`season`): 根据月份分类为 "Winter", "Spring", "Summer", "Autumn"。
   - 星期几 (`dayofweek`)
   - 小时 (`hour`，仅针对评论数据）
3. 计算发布后收到评论的时间间隔：
   - 使用 `datediff` 函数计算发布日期和评论日期之间的天数差。
4. 输出包含时间特征的新数据集。

**伪代码**:

```python
def extract_time_features(recipes_df, reviews_df):
    # 解析时间字段
    recipes_with_time = recipes_df.withColumn("DatePublished_parsed", to_timestamp(col("DatePublished")))
    reviews_with_time = reviews_df.withColumn("DateSubmitted_parsed", to_timestamp(col("DateSubmitted")))

    # 提取时间特征
    recipes_with_time = recipes_with_time \
        .withColumn("publish_year", year(col("DatePublished_parsed"))) \
        .withColumn("publish_month", month(col("DatePublished_parsed"))) \
        .withColumn("publish_season", when(...))  # 季节分类逻辑

    # 合并评论时间特征
    recipe_review_time = recipes_with_time.join(reviews_with_time, on="RecipeId") \
        .withColumn("days_to_review", datediff(col("DateSubmitted_parsed"), col("DatePublished_parsed")))

    return recipes_with_time, reviews_with_time, recipe_review_time
```

#### **2. 文本特征提取**

- **目标**: 对食谱的描述 (`Description`) 和关键词 (`Keywords`) 进行清理、分词、去停用词，并通过 TF-IDF 方法生成文本向量。
- **输入**: 食谱数据集中的 `Description` 和 `Keywords` 字段。
- **输出**: 包含文本特征（如 TF-IDF 向量、关键词数量、单词计数等）的新数据集。

**算法步骤**:

1. 清理文本：
   - 移除特殊字符、转换为小写。
   - 处理关键词字段的数组格式或字符串格式。
2. 分词和去停用词：
   - 使用 Spark 的 `Tokenizer` 和 `StopWordsRemover` 对文本进行分词和去停用词。
3. 构建 TF-IDF 特征：
   - 使用 `HashingTF` 和 `IDF` 计算文本的 TF-IDF 向量。
4. 提取统计特征：
   - 关键词数量 (`keyword_count`)
   - 描述单词数量 (`description_word_count`)
   - 描述字符数量 (`description_char_count`)
5. 输出包含文本特征的新数据集。

**伪代码**:

```python
def extract_text_features(recipes_df):
    # 文本清理
    recipes_cleaned = recipes_df \
        .withColumn("description_clean", clean_text_udf(col("Description"))) \
        .withColumn("keywords_processed", process_keywords_udf(col("Keywords")))

    # 分词和去停用词
    tokenizer = Tokenizer(inputCol="description_clean", outputCol="desc_tokens")
    stop_remover = StopWordsRemover(inputCol="desc_tokens", outputCol="desc_filtered")

    # TF-IDF 特征
    hashing_tf = HashingTF(inputCol="desc_filtered", outputCol="desc_tf", numFeatures=10000)
    idf = IDF(inputCol="desc_tf", outputCol="desc_tfidf")

    pipeline = Pipeline(stages=[tokenizer, stop_remover, hashing_tf, idf])
    model = pipeline.fit(recipes_cleaned)
    recipes_with_tfidf = model.transform(recipes_cleaned)

    # 统计特征
    recipes_with_features = recipes_with_tfidf \
        .withColumn("keyword_count", size(split(col("keywords_processed"), r"\s+"))) \
        .withColumn("description_word_count", size(col("desc_tokens"))) \
        .withColumn("description_char_count", length(col("description_clean")))

    return recipes_with_features
```

#### **3. 营养密度计算**

- **目标**: 基于食谱的营养成分（如卡路里、脂肪、蛋白质等），计算营养密度指标和评分。
- **输入**: 食谱数据集中的营养成分字段（如 `Calories`, `FatContent`, `ProteinContent` 等）。
- **输出**: 包含营养密度指标（如脂肪比例、蛋白质密度、钠含量比等）和综合评分的新数据集。

**算法步骤**:

1. 数据清洗和转换：
   - 将营养成分字段转换为数值类型，处理缺失值和异常值。
2. 计算营养密度指标：
   - 总宏量营养素 (`total_macros`) = 脂肪 + 碳水化合物 + 蛋白质
   - 脂肪比例 (`fat_ratio`) = 脂肪 / 总宏量营养素
   - 蛋白质密度 (`protein_density`) = 蛋白质 / 卡路里 \* 100
   - 纤维密度 (`fiber_density`) = 纤维 / 卡路里 \* 1000
3. 计算综合营养评分：
   - 权重分配：蛋白质密度 (30%)、纤维密度 (30%)、低钠 (20%)、低饱和脂肪 (20%)。
4. 输出包含营养密度指标和评分的新数据集。

**伪代码**:

```python
def calculate_nutrition_density(recipes_df):
    # 数据清洗和转换
    for col_name in ["Calories", "FatContent", "ProteinContent"]:
        recipes_df = recipes_df.withColumn(f"{col_name}_numeric", regexp_replace(col(col_name), r"[^\d.]", "").cast("double"))

    # 计算营养密度指标
    recipes_df = recipes_df \
        .withColumn("total_macros", col("FatContent_numeric") + col("CarbohydrateContent_numeric") + col("ProteinContent_numeric")) \
        .withColumn("fat_ratio", col("FatContent_numeric") / col("total_macros")) \
        .withColumn("protein_density", col("ProteinContent_numeric") / col("Calories_numeric") * 100)

    # 综合营养评分
    recipes_df = recipes_df.withColumn(
        "nutrition_score",
        col("protein_density") * 0.3 + col("fiber_density") * 0.3 + ...
    )

    return recipes_df
```

#### **4. 复杂度指标计算**

- **目标**: 基于食材数量、烹饪步骤数量和总时间，评估食谱的复杂度。
- **输入**: 食谱数据集中的 `RecipeIngredientParts` 和 `RecipeInstructions` 字段。
- **输出**: 包含复杂度指标（如食材数量、步骤数量、时间复杂度评分等）的新数据集。

**算法步骤**:

1. 统计食材数量和步骤数量：
   - 解析 `RecipeIngredientParts` 和 `RecipeInstructions` 字段，计算其长度。
2. 解析烹饪时间：
   - 将 `PrepTime`, `CookTime`, `TotalTime` 转换为分钟数。
3. 计算复杂度评分：
   - 时间复杂度评分：基于总时间的范围分配分数。
   - 食材复杂度评分：基于食材数量的范围分配分数。
   - 步骤复杂度评分：基于步骤数量的范围分配分数。
4. 综合复杂度评分：
   - 加权平均时间、食材和步骤复杂度评分。
5. 输出包含复杂度指标的新数据集。

**伪代码**:

```python
def calculate_complexity_score(recipes_df):
    # 统计食材数量和步骤数量
    recipes_df = recipes_df \
        .withColumn("ingredient_count", count_ingredients_udf(col("RecipeIngredientParts"))) \
        .withColumn("instruction_count", count_instructions_udf(col("RecipeInstructions")))

    # 解析烹饪时间
    recipes_df = recipes_df \
        .withColumn("prep_time_minutes", parse_time_udf(col("PrepTime"))) \
        .withColumn("cook_time_minutes", parse_time_udf(col("CookTime")))

    # 计算复杂度评分
    recipes_df = recipes_df \
        .withColumn("time_complexity_score", when(...)) \
        .withColumn("ingredient_complexity_score", when(...)) \
        .withColumn("overall_complexity_score", (col("time_complexity_score") + ...) / 3.0)

    return recipes_df
```
