# python3 analyze_ml1m.py \
#   --data-dir /home/comp/cswjyu/data/ml-1m \
#   --user-seqlen-plot ./painting/ml1m_user_seq_len.png \
#   --item-cdf-plot ./painting/ml1m_item_pop_cdf.png \
#   --summary-json ./painting/ml1m_summary.json \
#   --show-progress

python3 analyze_ml1m.py \
  --data-dir /home/comp/cswjyu/data/ml-20m \
  --user-seqlen-plot ./painting/ml20m_user_seq_len.png \
  --item-cdf-plot ./painting/ml20m_item_pop_cdf.png \
  --summary-json ./painting/ml20m_summary.json \
  --show-progress

# 用户聚类分析 - 分析MovieLens数据集中用户之间的item重合度和聚类效应
python3 analyze_ml_user_clustering.py \
  --data-dir /home/comp/cswjyu/data/ml-1m \
  --output-dir ./painting \
  --sample-users 6040 \
  --min-interactions 5 \
  --n-clusters 20 \
  --clustering-method kmeans \
  --similarity-metric jaccard \
  --max-heatmap-users 500 \
  --skip-silhouette \
  --show-progress  2>&1 | tee logs/ml1m-clustering.log

# ML-20M 用户聚类分析 - 推荐配置（平衡性能和结果质量）
# 方案1: 中等规模（推荐）- 约50GB内存，计算时间适中
python3 analyze_ml_user_clustering.py \
  --data-dir /home/comp/cswjyu/data/ml-20m \
  --output-dir ./painting \
  --sample-users 138000 \
  --min-interactions 5 \
  --n-clusters 20 \
  --clustering-method kmeans \
  --similarity-metric jaccard \
  --max-heatmap-users 500 \
  --skip-silhouette \
  --show-progress  2>&1 | tee logs/ml20m-clustering-medium.log

# 方案2: 大规模（需要大量内存）- 约152GB内存，计算时间很长
# python3 analyze_ml_user_clustering.py \
#   --data-dir /home/comp/cswjyu/data/ml-20m \
#   --output-dir ./painting \
#   --sample-users 138000 \
#   --min-interactions 5 \
#   --n-clusters 20 \
#   --clustering-method kmeans \
#   --similarity-metric jaccard \
#   --max-heatmap-users 500 \
#   --skip-silhouette \
#   --show-progress  2>&1 | tee logs/ml20m-clustering-large.log

# 方案3: 快速测试（小规模）- 约5GB内存，快速验证
python3 analyze_ml_user_clustering.py \
  --data-dir /home/comp/cswjyu/data/ml-20m \
  --output-dir ./painting \
  --sample-users 20000 \
  --min-interactions 5 \
  --n-clusters 50 \
  --clustering-method kmeans \
  --similarity-metric jaccard \
  --max-heatmap-users 500 \
  --skip-silhouette \
  --show-progress  2>&1 | tee logs/ml20m-clustering-small2.log


# 分析不同时间窗口下用户交互序列长度分布和活跃用户数
# ML-1M 数据集

# 30分钟窗口
python3 analyze_ml1m_time_window.py \
  --data-dir /home/comp/cswjyu/data/ml-1m \
  --window-type 30min \
  --show-progress \
  2>&1 | tee logs/ml1m-time-window-30min.log

# 1小时窗口
python3 analyze_ml1m_time_window.py \
  --data-dir /home/comp/cswjyu/data/ml-1m \
  --window-type 1h \
  --show-progress \
  2>&1 | tee logs/ml1m-time-window-1h.log

# 12小时窗口（半天）
python3 analyze_ml1m_time_window.py \
  --data-dir /home/comp/cswjyu/data/ml-1m \
  --window-type 12h \
  --show-progress \
  2>&1 | tee logs/ml1m-time-window-12h.log

# 1天窗口
python3 analyze_ml1m_time_window.py \
  --data-dir /home/comp/cswjyu/data/ml-1m \
  --window-type 1d \
  --show-progress \
  2>&1 | tee logs/ml1m-time-window-1d.log

# ============================================================================
# ML-20M 数据集分析（适用于更大的数据集）
# ============================================================================

# 30分钟窗口
python3 analyze_ml1m_time_window.py \
  --data-dir /home/comp/cswjyu/data/ml-20m \
  --window-type 30min \
  --show-progress \
  2>&1 | tee logs/ml20m-time-window-30min.log

# 1小时窗口
python3 analyze_ml1m_time_window.py \
  --data-dir /home/comp/cswjyu/data/ml-20m \
  --window-type 1h \
  --show-progress \
  2>&1 | tee logs/ml20m-time-window-1h.log

# 12小时窗口（半天）
python3 analyze_ml1m_time_window.py \
  --data-dir /home/comp/cswjyu/data/ml-20m \
  --window-type 12h \
  --show-progress \
  2>&1 | tee logs/ml20m-time-window-12h.log

# 1天窗口（推荐，因为ml-20m数据量大，使用1天窗口可以减少计算量）
python3 analyze_ml1m_time_window.py \
  --data-dir /home/comp/cswjyu/data/ml-20m \
  --window-type 1d \
  --show-progress \
  2>&1 | tee logs/ml20m-time-window-1d.log
