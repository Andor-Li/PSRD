import os
from loss import char_sam_loss

def create_generator(opt):
    # Use method_name (default 'sit') to select the generator method
    method = opt.method_name.lower()
    if opt.flag == 'train':
        if method == 'psrd':
            from model.PSRD import PSRD
            generator = PSRD(
                in_channels=opt.in_channels,
                dim=opt.dim,
                num_enc1_blocks=opt.num_blocks,
                num_enc2_blocks=opt.num_blocks,
                num_bottleneck_blocks=opt.num_blocks,
                num_dec_blocks=opt.num_blocks,
                num_refine_blocks=1,
                residual_scale=opt.residual_scale,
                init_alpha = opt.init_alpha
            )
            loss_function = char_sam_loss.PhysicsInformedLoss(lambda_sam=opt.lambda_sam)

        else:
            raise NotImplementedError(f"Generator method '{opt.method_name}' is not implemented.")      

        return generator, loss_function
    
    elif opt.flag == 'test':
        if method == 'psrd':
            from model.PSRD import PSRD
            generator = PSRD(
                in_channels=opt.in_channels,
                dim=opt.dim,
                num_enc1_blocks=opt.num_blocks,
                num_enc2_blocks=opt.num_blocks,
                num_bottleneck_blocks=opt.num_blocks,
                num_dec_blocks=opt.num_blocks,
                num_refine_blocks=1,
                residual_scale=opt.residual_scale,
                init_alpha = opt.init_alpha
            )
            
        else:
            raise NotImplementedError(f"Generator method '{opt.method_name}' is not implemented.")      

        return generator

    else:
        raise ValueError(f"Invalid flag '{opt.flag}'. Expected 'train' or 'test'.")

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_tag(opt):
    """Canonical run name: ``{method}_dim{dim}_alpha_{aa}_{dataset}``.

    ``alpha`` is the initial mixing coefficient of the hybrid normalization in
    PG-AFPG, encoded with two digits (0.4 -> ``04``, 1.0 -> ``10``).  ``round``
    is used instead of ``int`` because ``int(0.7 * 10) == 6`` in floating point.
    """
    return (f"{opt.method_name}_dim{opt.dim}"
            f"_alpha_{int(round(opt.init_alpha * 10)):02d}_{opt.dataset_name}")