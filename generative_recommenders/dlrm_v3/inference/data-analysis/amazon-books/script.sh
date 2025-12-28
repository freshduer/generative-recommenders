# 指定数据路径和输出目录
python analyze_amazon_dataset.py \
    --data_path ~/data/amzn_books/sasrec_format.csv \
    --output_dir ./reports

# 指定数据路径和输出目录
python analyze_user_repeat_visits.py \
    --data_path ~/data/amzn_books/sasrec_format.csv \
    --output_dir ./reports

# 分析热门用户占比
python analyze_top_users.py