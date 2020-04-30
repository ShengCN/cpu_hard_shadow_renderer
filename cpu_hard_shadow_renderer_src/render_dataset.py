import os
import multiprocessing
from functools import partial
from tqdm import tqdm
import argparse

def worker(input_param):
    CUDA, model, output_folder = input_param
    os.system('{} build/hard_shadow {} {}'.format(CUDA, model, output_folder))

    
def base_compute(param):
    x, y, shadow_list = param
    ret_np = np.zeros((256,256))
    for shadow_path in shadow_list:
        ret_np += 1.0 - plt.imread(shadow_path)[:,:,0]

    return x,y, ret_np

def multithreading_post_process(folder, output_folder, base_size=16):
    path = folder
    # output_folder = os.path.join(path, 'base')
    os.makedirs(output_folder, exist_ok=True)
    gt_file = os.path.join(path, 'ground_truth.txt')
    lines = []

    with open(gt_file) as f:
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            lines.append(r)

    print('there are {} lines'.format(len(lines)))

    group_data = {}
#     import pdb; pdb.set_trace()
    ibl_y_pos = []
    for l in tqdm(lines):
        if len(l) != 13:
            continue
            
        prefix = l[0]
        ibl = (int(l[1]), int(l[2]))
        ibl_y_pos.append(ibl[1])
        camera_pos = (l[3], l[4], l[5])
        rot = l[6]
        target_center = (l[7], l[8], l[9])
        light_pos = (l[10], l[11], l[12])

        key = (camera_pos, rot)
        if key not in group_data.keys():
            group_data[key] = dict()

        ibl_key = ibl
        group_data[key][ibl_key] = prefix

    print('keys: ', len(group_data.keys()))
    print('keys: ', group_data.keys())
    # img_folder = os.path.join(path, 'imgs')
    img_folder = path
    x_begin, y_begin = 0, min(ibl_y_pos)

    for key_id, key in enumerate(group_data.keys()):
        # prepare mask
        prefix = group_data[key][(x_begin,y_begin)]
#         mask_np = plt.imread(os.path.join(img_folder, '{}_mask.png'.format(prefix)))
#         mask_output = os.path.join(output_folder, '{:03d}_mask.npy'.format(key_id))
#         np.save(mask_output, mask_np[:,:,0])

        # prepare shadow
        input_list = []
        x_iter, y_iter = 512//base_size, (256-y_begin) // base_size
        group_np = np.zeros((256,256, x_iter, y_iter))
        for xi in tqdm(range(x_iter)):
            for yi in range(y_iter):
               # share all shadow results
                tuple_input = [xi, yi]
                shaodw_list = [os.path.join(img_folder,
                                            '{}_shadow.png'.format(group_data[key][(xi * base_size + i, y_begin + yi * base_size + j)]))
                               for i in range(base_size)
                               for j in range(base_size)]
                tuple_input.append(shaodw_list)
                input_list.append(tuple_input)
        processer_num, task_num = 128, len(input_list)
        base_weight = 1.0 / (base_size * base_size)
        with multiprocessing.Pool(processer_num) as pool:
            for i, base in enumerate(pool.imap_unordered(base_compute, input_list), 1):
                x,y, base_np = base[0], base[1], base[2]
                group_np[:,:,x,y] = base_np * base_weight
                print("Finished: {} \r".format(float(i) / task_num), flush=True, end='')

        output_path = os.path.join(output_folder, '{:03d}_shadow.npy'.format(key_id))
        np.save(output_path, group_np)
        del group_np

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('proc', default=1,type=int, help='how many processing')
    args = parser.parse_args()

    print('{} processing is using'.format(args.proc))

    model_folder = '../models/'
    model_files = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, f))]
    print('There are {} model files'.format(len(model_files)))
    
    cur_folder = os.path.abspath('.')
    output_list,base_output_list = [], []
    ds_root = './output'
    base_ds_root = './base/'
    os.makedirs(ds_root, exist_ok=True)
    graphics_card = []
    for i, f in tqdm(enumerate(model_files)):
        card = 0;
        graphics_card.append('CUDA_VISIBLE_DEVICES={}'.format(card))
        
        model_fname = os.path.splitext(os.path.basename(f))[0]
        # out_folder = os.path.join(os.path.join(cur_folder, 'output'), model_fname)
        # os.makedirs(out_folder, exist_ok=True)
        out_folder = os.path.join(ds_root, model_fname)
        output_list.append(out_folder)
        
        base_output_folder = os.path.join(base_ds_root, model_fname)
        base_output_list.append(base_output_folder)
    
    input_param = zip(graphics_card, model_files, output_list)
    # processor_num = len(model_files)
    processor_num = args.proc
    total = len(model_files)
    with multiprocessing.Pool(processor_num) as pool:
        for i,_ in enumerate(pool.imap_unordered(worker, input_param), 1):
            print('Finished: {} \r'.format(float(i)/total), flush=True, end='')
    
    print('begin preparing bases')
    for i, shadow_output_folder in tqdm(enumerate(output_list)):
        multithreading_post_process(shadow_output_folder, base_output_folder[i])
        
    print('Dataset generation finished')
    print('Bases are in folder ./base')
    
