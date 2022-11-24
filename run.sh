for i in {2..25}
do
    python src/main.py --n_agents 0 --n_others $i --num_sub_slot 200 --aloha_prob 0.5 --env_mac_mode 0 --max_iter 10000 --save_trace --save_track --setup_path setups/ --mask mask --delay delay --x x --y y --z z --log_path logs/ --config_path configs/ --track_path tracks/ --file_prefix mobile_ALOHA_ > temp/exp$i &
done