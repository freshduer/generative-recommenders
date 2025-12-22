# 指定数据路径和输出目录
python analyze_amazon_dataset.py \
    --data_path ~/data/amzn_books/sasrec_format.csv \
    --output_dir ./reports

# 分析不同时间窗口的序列长度
# 分析单个时间窗口（例如：1天）
python analyze_sequence_length_by_time_window.py \
    --data_path ~/data/amzn_books/sasrec_format.csv \
    --window_type 1d \
    --show_progress

# 分析所有时间窗口（30分钟、1小时、半天、一天）
python analyze_sequence_length_by_time_window.py \
    --data_path ~/data/amzn_books/sasrec_format.csv \
    --all_windows \
    --show_progress

# 指定输出目录
python analyze_sequence_length_by_time_window.py \
    --data_path ~/data/amzn_books/sasrec_format.csv \
    --window_type 1h \
    --output_dir ./reports \
    --show_progress