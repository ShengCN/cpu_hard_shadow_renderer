make -j8
sudo rm airplane/*shadow.png
sudo nvprof ./hard_shadow --model="/home/ysheng/Dataset/general_models/vase_0017.off" --output="airplane" --cam_pitch=30 --model_rot=-45 --render_touch
