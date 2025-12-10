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
