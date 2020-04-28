import os
import multiprocessing
from functools import partial
from tqdm import tqdm

def worker(input_param):
    model, output_folder = input_param
    os.system('build/hard_shadow {} {}'.format(model, output_folder))

if __name__ == '__main__':
    model_folder = '../models/'
    model_files = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, f))]
    print('There are {} model files'.format(len(model_files)))
    
    cur_folder = os.path.abspath('.')
    output_list = []
    for f in tqdm(model_files):
        model_fname = os.path.splitext(os.path.basename(f))[0]
        out_folder = os.path.join(os.path.join(cur_folder, 'output'), model_fname)
        os.makedirs(out_folder, exist_ok=True)
        
        output_list.append(out_folder)
        
    input_param = zip(model_files, output_list)
    # processor_num = len(model_files)
    processor_num = 1
    total = len(model_files)
    with multiprocessing.Pool(processor_num) as pool:
        for i,_ in enumerate(pool.imap_unordered(worker, input_param), 1):
            print('Finished: {} \r'.format(float(i)/total), flush=True, end='')
    
    print('Dataset generation finished')