for i in {10..25}
do
    python src/main.py --n_agents 0 --n_others $i --movable --save_trace --save_track --log_path logs/ --config_path configs/ --track_path tracks/ --file_prefix mobile_ALOHA_ > temp/exp$i &
done