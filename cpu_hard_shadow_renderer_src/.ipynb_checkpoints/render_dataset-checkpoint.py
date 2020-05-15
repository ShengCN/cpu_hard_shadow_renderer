import os
from os.path import join
import multiprocessing
from functools import partial
from tqdm import tqdm
import argparse
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

def get_newest_prefix(out_folder):
    files = [f for f in os.listdir(out_folder) if (os.path.isfile(os.path.join(out_folder, f))) and (f.find('_shadow')!=-1)]
    return len(files) - 1

def worker(input_param):
    model, output_folder, gpu, resume, cam_pitch, model_rot = input_param
    os.makedirs(output_folder, exist_ok=True)
    
    newest_prefix = get_newest_prefix(output_folder)
    os.system('build/hard_shadow --model={} --output={} --gpu={} --resume={} --cam_pitch={} --model_rot={}'.format(model, output_folder, gpu, resume,cam_pitch,model_rot))
              
def base_compute(param):
    x, y, shadow_list = param
    ret_np = np.zeros((256,256))
    for shadow_path in shadow_list:
        ret_np += 1.0 - plt.imread(shadow_path)[:,:,0]

    return x,y, ret_np

def multithreading_post_process(folder, output_folder, base_size=16):
    def get_ibl_i_j(path):
        """ Given a path, return (i,j)
        """
        basename = os.path.basename(path)
        ibli_, iblj_ = 'ibli_','iblj_'
        i = basename[basename.find(ibli_) + len(ibli_):basename.find('_iblj')]
        j = basename[basename.find(iblj_) + len(iblj_):basename.find('_shadow')]
        return i,j
    
    os.makedirs(output_folder, exist_ok=True)
    files = [join(folder, f) for f in os.listdir(folder) if os.path.isfile(join(folder, f))]
    i_list, j_list = set(),set()
    pitch_rot_list = set()
    
    for f in tqdm(files):
        i,j = get_ibl_i_j(f)
        i_list.add(int(i))
        j_list.add(int(j))
        pitch_rot = f[:f.find('ibli')-1]
        pitch_rot_list.add(pitch_rot)
    
    pitch_rot_list = list(pitch_rot_list)
    i_list, j_list = list(i_list), list(j_list)
    i_list.sort()
    j_list.sort()
    
    max_i, min_i = i_list[-1], i_list[0]
    min_j, max_j, j_diff = j_list[0], j_list[-1], j_list[1]-j_list[0]    
    print(max_i, min_i, max_j, min_j)
    
    x_iter, y_iter = (max_i-min_i)//base_size,(max_j-min_j)//base_size
    base_iter = base_size//j_diff
    print(x_iter, y_iter)
    
    group_np = np.zeros((256,256, x_iter, y_iter))
    input_list = []
    for pitch_rot in pitch_rot_list:
        for xi in tqdm(range(x_iter)):
            for yi in range(y_iter):
                tuple_input = [xi, yi]
                
                shaodw_list = [os.path.join('{}_ibli_{}_iblj_{}_shadow.png'.format(pitch_rot, min_i + xi * base_size + i * j_diff, min_j + yi * base_size + j * j_diff))
                               for i in range(base_iter)
                               for j in range(base_iter)]
                tuple_input.append(shaodw_list)
                input_list.append(tuple_input)

        processer_num, task_num = 1, len(input_list)
        base_weight = 1.0 / (base_iter * base_iter)
        with multiprocessing.Pool(processer_num) as pool:
            for i, base in enumerate(pool.imap_unordered(base_compute, input_list), 1):
                x,y, base_np = base[0], base[1], base[2]
                group_np[:,:,x,y] = base_np * base_weight
                print("Finished: {} \r".format(float(i) / task_num), flush=True, end='')
        
        base_pitch_rot = os.path.basename(pitch_rot)
        output_path = os.path.join(output_folder, '{}_shadow.npy'.format(base_pitch_rot))
        np.save(output_path, group_np)
    del group_np
        
def render(args, model_files):
    cur_folder = os.path.abspath('.')
    output_list,base_output_list = [], []
    cache_folder = './cache'
    
    ds_root = os.path.join(cache_folder, 'output')
    base_ds_root = './base/'
        
    os.makedirs(cache_folder, exist_ok=True)
    os.makedirs(ds_root, exist_ok=True)
    os.makedirs(base_ds_root, exist_ok=True)
    graphics_card = [args.gpu] * len(model_files)
    
    if args.resume:
        resume_arg = 'true'
    else:
        resume_arg = 'false'
    resume_list = [resume_arg] * len(model_files)
              
#     model, output_folder, gpu, resume, cam_pitch, model_rot
    
    cam_pitch = [args.cam_pitch] * len(model_files)
    model_rot = [args.model_rot] * len(model_files)
              
    for i, f in tqdm(enumerate(model_files)):        
        model_fname = os.path.splitext(os.path.basename(f))[0]
        out_folder = os.path.join(ds_root, model_fname)
        output_list.append(out_folder)
        
        base_output_folder = os.path.join(base_ds_root, model_fname)
        base_output_list.append(base_output_folder)
    
    input_param = zip(model_files, output_list, graphics_card, resume_list, cam_pitch, model_rot)
    total = len(model_files)
    processor_num = 1
    with multiprocessing.Pool(processor_num) as pool:
        for i,_ in enumerate(pool.imap_unordered(worker, input_param), 1):
            print('Finished: {} \r'.format(float(i)/total), flush=True, end='')
    
    print('begin preparing bases')
    for i, shadow_output_folder in tqdm(enumerate(output_list)):
        print('shadow output: {}, base output: {}'.format(shadow_output_folder, base_output_list[i]))
        multithreading_post_process(shadow_output_folder, base_output_list[i])
        
    print('Bases render finish, check folder {}'.format(base_ds_root))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', default=0, type=int, help='GPU device')
    parser.add_argument('--start_id', default=0, type=int, help='Start file id, range [0, 65]')
    parser.add_argument('--end_id', default=66, type=int, help='End file id, range [0, 65]')
    parser.add_argument("--resume", help="skip the rendered image", action="store_true")
    parser.add_argument("--cam_pitch", type=str,help="list of camera pitch")
    parser.add_argument("--model_rot", type=str, help="list of model rotation")
    
    args = parser.parse_args()

    print('parameters: {}'.format(args))

    model_folder = '../simplified_models/'
    model_files = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, f))]
    print('There are {} model files'.format(len(model_files)))
    model_files.sort()
    model_files = model_files[args.start_id : args.end_id+1]
    
    print('Will render {} files'.format(len(model_files)))
    begin = time.time()
    render(args, model_files)
    elapsed = time.time() - begin
    print("Total time: {} mins".format(elapsed/60.0))
