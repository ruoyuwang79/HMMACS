for i in {2..25}
do
    python src/main.py --n_agents 0 --n_others $i --num_sub_slot 20 --tdma_occupancy 5 --aloha_prob 0.5 --env_mode 1 --env_mac_mode 0 --agent_mac_mode 0 --max_iter 50000 --setup_path setups/ --mask agent_aloha_mask --log_path logs/ --config_path configs/ --track_path tracks/ --file_prefix diff_sync_static_ --save_trace > temp/exp$i &
done