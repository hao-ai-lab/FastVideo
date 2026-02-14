import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


def general_ocr_sd3_rationorm():
    
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.run_name = "XXX"
    config.run_project = "XXXX"

    config.resolution = 512
    gpu_number = 16
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.0
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = False
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = f'checkpoints/logs/ocr/{config.run_name}'
    config.reward_fn = {
        "ocr": 1.0,
        "pickscore": 0.0
    }
    
    config.prompt_fn = "general_ocr"
    config.mixed_precision = "fp16"

    config.train.clip_range=2e-6
    config.train.highclip_range=2e-6
    config.train.learning_rate = 1e-4
    config.rationorm = True

    config.per_prompt_stat_tracking = True
    return config

def pickscore_hps_sd3_ratio_norm():
    
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.run_name = "XXX"
    config.run_project = "XXXX"

    config.resolution = 512
    gpu_number=16
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.0
    config.sample.global_std = False
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = f'checkpoints/logs/pickscore/{config.run_name}'
    config.reward_fn = {
        "pickscore": 1.0
    }
    
    config.prompt_fn = "general_ocr"
    config.mixed_precision = "fp16"
    config.train.clip_range=4e-6
    config.train.highclip_range=4e-6
    config.train.learning_rate = 1e-4
    config.rationorm = True
    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_rationorm():
    gpu_number = 16
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.resolution = 512

    config.run_name = "XXX"
    config.run_project = "XXXX"
    
    gpu_number=16
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.0
    config.sample.global_std = False
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    
    config.reward_fn = {
        "geneval": 1.0,
        "pickscore": 0.0
    }
    
    config.prompt_fn = "geneval"
    config.train_attribute = None # select from ['color_attr','colors','counting','single_object','two_object']
    config.mixed_precision = "fp16"
    config.train.clip_range=2e-6
    config.train.highclip_range=2e-6
    config.train.learning_rate = 1e-4
    config.rationorm = True

    config.save_dir = f'checkpoints/logs/geneval/{config.run_name}'

    config.per_prompt_stat_tracking = True
    return config

def get_config(name):
    return globals()[name]()
