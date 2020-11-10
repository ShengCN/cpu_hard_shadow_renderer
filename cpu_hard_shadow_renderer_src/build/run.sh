make -j8
sudo rm airplane/*shadow.png
sudo nvprof ./hard_shadow --model="/home/ysheng/Dataset/benchmark_ds/models/general/bottle_0344.off" --output="airplane" --cam_pitch=30 --model_rot=-45 --render_touch --render_mask
