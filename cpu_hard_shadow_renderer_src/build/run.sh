make -j8
sudo rm airplane/*shadow.png
sudo nvprof ./hard_shadow --model="/home/ysheng/Dataset/human_models/simulated_combine_male_short_outfits_genesis8_armani_sweateroutfitall_Base_Pose_Walking_B.obj" --output="airplane" --cam_pitch=30 --model_rot=-45 --render_touch --render_mask
