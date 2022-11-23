for i in {2..25}
do
    python src/main.py --n_agents 0 --n_others $i --num_sub_slot 20 --env_mac_mode 1 --movable --max_iter 50000 --save_trace --save_track --log_path logs/ --config_path configs/ --track_path tracks/ --file_prefix mobile_ALOHA_ > temp/exp$i &
done