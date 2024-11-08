

# Start the training job in a detached tmux session
tmux new-session -d -s "bench_job" '
 export CUDA_VISIBLE_DEVICES=1,2,3,7
  eval "$(conda shell.bash hook)"
  conda activate finenv
  
   python test_all.py

  read -p "Press Enter to exit..."
'


tmux attach -t "bench_job"
