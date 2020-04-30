import os
import multiprocessing
from functools import partial
from tqdm import tqdm
import argparse

def worker(input_param):
    CUDA, model, output_folder = input_param
    os.system('{} build/hard_shadow {} {}'.format(CUDA, model, output_folder))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('proc', default=1,type=int, help='how many processing')
    args = parser.parse_args()

    print('{} processing is using'.format(args.proc))

    model_folder = '../models/'
    model_files = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, f))]
    print('There are {} model files'.format(len(model_files)))
    
    cur_folder = os.path.abspath('.')
    output_list = []
    ds_root = '/home/ysheng/Dataset/new_dataset'
    graphics_card = []
    for i, f in tqdm(enumerate(model_files)):
        card = i % 3;
        graphics_card.append('CUDA_VISIBLE_DEVICES={}'.format(card))
        
        model_fname = os.path.splitext(os.path.basename(f))[0]
        # out_folder = os.path.join(os.path.join(cur_folder, 'output'), model_fname)
        # os.makedirs(out_folder, exist_ok=True)
        out_folder = os.path.join(ds_root, model_fname)
        output_list.append(out_folder)
    
    input_param = zip(graphics_card, model_files, output_list)
    # processor_num = len(model_files)
    processor_num = args.proc
    total = len(model_files)
    with multiprocessing.Pool(processor_num) as pool:
        for i,_ in enumerate(pool.imap_unordered(worker, input_param), 1):
            print('Finished: {} \r'.format(float(i)/total), flush=True, end='')
    
    print('Dataset generation finished')
