from argparse import Namespace

def build_predrnn_args(img_width, data_dir, result_checkpoint_dir, dataset_name='latent'):
    return Namespace(
        is_training=0,
        device='cuda',
        dataset_name=dataset_name,
        train_data_paths=data_dir,
        valid_data_paths=data_dir,
        save_dir=f'checkpoints/{result_checkpoint_dir}',
        gen_frm_dir=f'results/{result_checkpoint_dir}',
        model_name='predrnn_v2',
        visual=0,
        reverse_input=1,
        img_width=img_width,
        img_channel=1,
        input_length=10,
        total_length=20,
        num_hidden=f'{img_width}, {img_width}, {img_width}, {img_width}',
        filter_size=5,
        stride=1,
        patch_size=4,
        layer_norm=0,
        decouple_beta=0.01,
        reverse_scheduled_sampling=1,
        r_sampling_step_1=5000,
        r_sampling_step_2=50000,
        r_exp_alpha=2000,
        lr=0.0001,
        batch_size=4,
        max_iterations=2000,
        display_interval=100,
        snapshot_interval=100
    )
