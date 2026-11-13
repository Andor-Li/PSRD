import argparse
import model_ini
import tester

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Saving and loading parameters
    parser.add_argument('--flag', type=str, default='test', help='train or test')
    parser.add_argument('--save_test_folder', type=str, default='test_results', help='Folder to save test results')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size')
    # GPU parameters
    parser.add_argument('--multi_gpu', type=bool, default=False, help='Whether to use multiple GPUs')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Enable cudnn benchmark')
    # Testing parameters
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker threads for data loading')

    parser.add_argument('--gpu_id', type=int, default=0, help='IDs of GPU to use')

    parser.add_argument('--dataset_name', type=str, default='BJTU_31', help='Dataset name: BJTU_31 / BJTU_204 / BJTU_E_31 / BJTU_E_204')
    parser.add_argument('--merged_root', type=str, default='',
                        help='Root of the cached dataset. If empty, the default path hard-coded in tester.py is used')
    parser.add_argument('--in_channels', type=int, default=31, help='Number of input channels for the generator')
    parser.add_argument('--out_channels', type=int, default=31, help='Number of output channels for the generator')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
   
    parser.add_argument('--method_name', type=str, default='psrd', help='Name of the method to use')
    parser.add_argument('--dim', type=int, default=32, help='dim')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of blocks in the network')
    parser.add_argument('--init_alpha', type=float, default=0.4, help='Initial value for alpha in hybrid normalization of PG-AFPG')
    parser.add_argument('--residual_scale', type=float, default=0.1, help='Residual scale for the network')

    parser.add_argument('--model_path', type=str, default='',
                        help='Path to the checkpoint. Defaults to checkpoints/<method>_dim<dim>_alpha_<aa>_<dataset>.pth')

    opt = parser.parse_args()

    if not opt.model_path:
        opt.model_path = f'checkpoints/{model_ini.run_tag(opt)}.pth'

    tester.Tester(opt)