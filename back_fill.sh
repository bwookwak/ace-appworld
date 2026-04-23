for N in 50 100 150 200 250; do
  appworld run eval_ckpt_$N --num-processes 5 --root /home/bwoo/workspace/ace-appworld \
  && appworld evaluate eval_ckpt_$N test_normal --root /home/bwoo/workspace/ace-appworld
done