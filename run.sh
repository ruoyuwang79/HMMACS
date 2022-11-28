for i in {1..24}
do
    python src/main.py --n_agents 1 --n_others $i --num_sub_slot 20 --tdma_occupancy 2 --aloha_prob 0.2 --env_mode 1 --env_mac_mode 1 --agent_mac_mode 1 --max_iter 50000 --save_trace --save_track --setup_path setups/ --delay delay --x x --y y --z z --log_path logs/ --config_path configs/ --track_path tracks/ --file_prefix agent_coexist_ > temp/exp$i &
done