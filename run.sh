for i in {2..25}
do
    python src/main.py --n_agents 0 --n_others $i --num_sub_slot 20 --tdma_occupancy 5 --aloha_prob 0.5 --env_mac_mode 1 --agent_mac_mode 1 --max_iter 50000 --save_trace --save_track --setup_path setups/ --mask agent_aloha_mask --delay free_delay --x x --y y --z z --log_path logs/ --config_path configs/ --track_path tracks/ --file_prefix sync_ > temp/exp$i &
done