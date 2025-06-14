import torch
from imageio import imread, imsave, imwrite
from PIL import Image
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from model import UNet3Plus_DeepSup_CGM, DispNetS
from utils.comprehensive import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
# 设置其一
parser.add_argument("--use-eigen", default=True, type=bool, help="use kitti eigen list")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")

parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

'''
python run_inference.py --output-disp --pretrained ./checkpoints/KITTI-sfm\,epoch_size1000/09-22-09\:32/dispnet_model_best.pth.tar --dataset-dir ./depth_eva_image/sem_test/ --output-dir ./depth_eva_image/sem_test/
'''


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return
    
    print("=> loading model...")
    disp_net = DispNetS().to(device)  # UNet3Plus_DeepSup_CGM
    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.dataset_list is not None:
        print("=> using kitti eigen split dataset !")
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([list(dataset_dir.walkfiles('*.{}'.format(ext))) for ext in args.img_exts], [])

    print('=> {} files to test'.format(len(test_files)))


    p = 0
    for file in tqdm(test_files):
        p = p+1
        img = imread(file)

        h,w,_ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = np.array(Image.fromarray(img).resize((args.img_width, args.img_height)))  # imresize
        rawimg = img
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.5)/0.5).to(device)

        output = disp_net(tensor_img)#[0]

        file_path, file_ext = file.relpath(args.dataset_dir).splitext()
        #print(file_path)
        #print(file_path.splitall())
        file_name = '-'.join(file_path.splitall()[1:])
        #print(file_name)
        if False:  # 输出保存原始图像
            imsave('test_result/depth/kitti_eigen_rawimg/'+'{}{}'.format(file_name, file_ext), rawimg)
        if args.output_disp:  # magma rainbow bone
            disp = (255*tensor2array(output, max_value=None, colormap='magma')).astype(np.uint8)
            imsave(output_dir/'{}_disp{}'.format(file_name, file_ext), np.transpose(disp, (1,2,0)))

            #imsave(output_dir / '__{}_{}_img{}'.format(p, file_name, file_ext), np.transpose(img, (1, 2, 0)))

        if args.output_depth:
            depth = 1/output
            depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
            imsave(output_dir/'{}_depth{}'.format(file_name, file_ext), np.transpose(depth, (1,2,0)))
        print("=> the depth file saving on {}".format(output_dir))


if __name__ == '__main__':
    main()
