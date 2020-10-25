make -j8
sudo rm airplane/*shadow.png
sudo nvprof ./hard_shadow --model="/home/ysheng/Dataset/general_models/airplane_0415.off" --output="airplane" --cam_pitch=0 --model_rot=0,90,-90,45,-45 --render_mask --render_shadow --render_normal
