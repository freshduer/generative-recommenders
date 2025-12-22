# 1) 统计 KuaiRand-1K 数据概况并导出 CSV/JSON
python analyze_kuairand.py \
  --data-dir /home/comp/cswjyu/data/KuaiRand-27K/data \
  --summary-json reports/kuairand_summary.json \
  --user-stats-csv reports/user_interactions.csv \
  --item-stats-csv reports/item_popularity.csv \
  --user-unique-items-csv reports/user_unique_items.csv \
  --user-duplicate-surplus-csv reports/user_duplicate_surplus.csv \
  --user-top-item-csv reports/user_top_item_stats.csv \
  --stats-json reports/kuairand_stats.json \
  --duplicates-csv reports/duplicates.csv \
  --user-dedup-plot reports/user_sequence_dedup.png \
  --top-k-items 20 \
  --top-percent 10 \
  --show-progress

# 2) 画分布图 + 物品热度 CDF（带红色 90%/99% 虚线），并计算 Top 1% / Top 10% 占比
python plot_kuairand_summary.py \
  --summary-json reports/kuairand_summary.json \
  --out-dir reports \
  --item-stats-csv reports/item_popularity.csv \
  --cdf-item-pop

# 2.1) 画用户序列去重效果图
python plot_dedup_from_exports.py \
  --interactions-csv reports/user_interactions.csv \
  --duplicate-surplus-csv reports/user_duplicate_surplus.csv \
  --output reports/dedup_summary.png

# 3) 用户聚类分析 - 分析用户之间的item重合度和聚类效应
# 对于27K用户，使用以下优化参数：
# - 跳过轮廓系数计算以加速（或使用采样）
# - 使用K-means聚类（比层次聚类快）
# python analyze_user_clustering.py \
#   --data-dir /home/comp/cswjyu/data/KuaiRand-27K/data \
#   --output-dir reports \
#   --sample-users 27000 \
#   --min-interactions 5 \
#   --n-clusters 5 \
#   --clustering-method kmeans \
#   --similarity-metric jaccard \
#   --max-heatmap-users 500 \
#   --skip-silhouette \
#   --show-progress

#分析不同时间窗口下数据长度
# 按天分析（默认）
python analyze_sequence_length_by_time_window.py \
  --data-dir /home/comp/cswjyu/data/KuaiRand-27K/data \
  --window-type day \
  --output-dir reports \
  --show-progress

# 按周分析
python analyze_sequence_length_by_time_window.py \
  --data-dir /home/comp/cswjyu/data/KuaiRand-27K/data \
  --window-type week \
  --show-progress

# 按小时分析
python analyze_sequence_length_by_time_window.py \
  --data-dir /home/comp/cswjyu/data/KuaiRand-27K/data \
  --window-type hour \
  --show-progress
