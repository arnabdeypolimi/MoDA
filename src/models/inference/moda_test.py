# encoding = 'utf-8'
import os
import os.path as osp
import sys
from omegaconf import OmegaConf


import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))))

from src.datasets.preprocess.extract_features.audio_processer import AudioProcessor
from src.datasets.preprocess.extract_features.motion_processer import MotionProcesser
from src.models.dit.talking_head_diffusion import MotionDiffusion

from src.utils.logger import get_logger
import time

# Initialize module logger
logger = get_logger(__name__)

emo_map = {
    0: 'Anger', 
    1: 'Contempt', 
    2: 'Disgust', 
    3: 'Fear', 
    4: 'Happiness', 
    5: 'Neutral', 
    6: 'Sadness', 
    7: 'Surprise',
    8: 'None'
}
# import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭 CuDNN 优化以保证可复现性

# 在推理前调用
set_seed(42)

class NullableArgs:
    def __init__(self, namespace):
        for key, value in namespace.__dict__.items():
            setattr(self, key, value)


class LiveVASAPipeline(object):
    def __init__(self, cfg_path: str, load_motion_generator: bool = True, motion_mean_std_path=None):
        """The pipeline for LiveVASA
        The pipeline for LiveVASA

        Args:
            cfg_path (str): YAML config file path of LiveVASA
        """
        logger.info("=" * 60)
        logger.info("Initializing LiveVASA Pipeline")
        logger.info("=" * 60)
        
        # pretrained encoders of live portrait
        cfg = OmegaConf.load(cfg_path)
        self.device_id = cfg.device_id
        self.device = f"cuda:{self.device_id}"
        logger.info(f"Using device: {self.device}")
        
        # 1 load audio processor
        logger.info("Loading audio processor...")
        self.audio_processor: AudioProcessor = AudioProcessor(cfg_path=cfg.audio_model_config, is_training=False)
        logger.info("Audio processor loaded successfully")

        if cfg.motion_models_config is not None and load_motion_generator:
            logger.info("Loading motion generator...")
            motion_models_config = OmegaConf.load(cfg.motion_models_config)
            logger.debug(f"Motion models config loaded from: {osp.realpath(cfg.motion_models_config)}")
            self.motion_generator = MotionDiffusion(motion_models_config, device=self.device)
            self.load_motion_generator(self.motion_generator, cfg.motion_generator_path)
            logger.info("Motion generator loaded successfully")
        else:
            self.motion_generator = None    
            logger.warning("Motion generator not loaded (disabled by config)")
        
        # 3. load motion processer
        logger.info("Loading motion processor...")
        self.motion_processer: MotionProcesser = MotionProcesser(cfg_path=cfg.motion_processer_config, device_id=cfg.device_id)
        logger.info("Motion processor loaded successfully")


        self.motion_mean_std = None
        if motion_mean_std_path is not None:
            logger.debug(f"Loading motion mean/std from: {motion_mean_std_path}")
            self.motion_mean_std = torch.load(motion_mean_std_path)
            self.motion_mean_std["mean"] = self.motion_mean_std["mean"].to(self.device)
            self.motion_mean_std["std"] = self.motion_mean_std["std"].to(self.device)
            logger.debug(f"scale mean: {self.motion_mean_std['mean'][0, 63:64]}, std: {self.motion_mean_std['std'][0, 63:64]}")
            logger.debug(f"t mean: {self.motion_mean_std['mean'][0, 64:67]}, std: {self.motion_mean_std['std'][0, 64:67]}")
            logger.debug(f"pitch mean: {self.motion_mean_std['mean'][0, 67:68]}, std: {self.motion_mean_std['std'][0, 67:68]}")
            logger.debug(f"yaw mean: {self.motion_mean_std['mean'][0, 68:69]}, std: {self.motion_mean_std['std'][0, 68:69]}")
            logger.debug(f"roll mean: {self.motion_mean_std['mean'][0, 69:70]}, std: {self.motion_mean_std['std'][0, 69:70]}")

        self.cfg = cfg
        logger.info("LiveVASA Pipeline initialization complete")

    def set_motion_generator(self, motion_generator: MotionDiffusion):
        logger.debug("Setting custom motion generator")
        self.motion_generator = motion_generator
        self.motion_generator.to(self.device)
        
    def load_motion_generator(self, model, motion_generator_path: str):
        logger.info(f"Loading motion generator weights from: {motion_generator_path}")
        model_data = torch.load(motion_generator_path, map_location=self.device)
        model.load_state_dict(model_data, strict=False)
        logger.debug("Motion generator weights loaded")

        model.to(self.device)
        model.eval()
        logger.debug("Motion generator moved to device and set to eval mode")

    def modulate_lip(self, standard_motion: torch.Tensor, motions: torch.Tensor, alpha=5, beta=0.1):
        # standard_motion: 63
        # motions: Tx63
        standard_exp = standard_motion[:63].reshape(1, 21, 3)
        exps = motions[:, :63].reshape(-1, 21, 3)
        exp_deltas = exps - standard_exp

        # calc weights
        lip_deltas = []
        for lip_idx in [6, 12, 14, 17, 19, 20]:
            lip_deltas.append(exp_deltas[:, lip_idx, :])
        lip_deltas = torch.stack(lip_deltas, dim=1)   # T, 6, 3
        lip_deltas = lip_deltas.view(lip_deltas.shape[0], -1) 
        lip_dist = torch.sum(lip_deltas ** 2, dim=-1, keepdim=True)
        max_dist = torch.max(lip_dist, dim=0)[0].squeeze()   # 1
        weight = (torch.sigmoid(lip_dist*alpha) - 0.5) / (max_dist * beta + 0.05) 

        # modulation
        for lip_idx in [6, 12, 14, 17, 19, 20]:
            exps[:, lip_idx, :] = standard_exp[:, lip_idx, :] + exp_deltas[:, lip_idx, :] * (1 + weight)
        motions[:, :63] = exps.flatten(-2, -1)
        
        return motions
    
    def get_motion_sequence(self, motion_data: torch.Tensor, rescale_ratio=1.0):
        n_frames = motion_data.shape[0]
        # denorm
        if self.motion_mean_std is not None:
            if motion_data.shape[1] > 70:
                motion_data[:, :63] = motion_data[:, :63] * (self.motion_mean_std["std"][:, :63] + 1e-5) + self.motion_mean_std["mean"][:, :63]
                # denorm pose
                motion_data[:, 63:] = motion_data[:, 63:] + self.motion_mean_std["mean"][:, 63:]
            else:
                motion_data = motion_data * (self.motion_mean_std["std"] + 1e-5) + self.motion_mean_std["mean"]

        kp_infos = {"exp": [], "scale": [], "t": [], "pitch": [], "yaw": [], "roll": []}
        for idx in range(n_frames):
            exp = motion_data[idx][:63]
            scale = motion_data[idx][63:64] * rescale_ratio
            t = motion_data[idx][64:67] * rescale_ratio
            if motion_data.shape[1] > 70:
                pitch = motion_data[idx][67:133]
                yaw = motion_data[idx][133:199]
                roll = motion_data[idx][199:265]
            else:
                pitch = motion_data[idx][67:68]
                yaw = motion_data[idx][68:69]
                roll = motion_data[idx][69:70]

            kp_infos["exp"].append(exp)
            kp_infos["scale"].append(scale)
            kp_infos["t"].append(t)
            kp_infos["pitch"].append(pitch)
            kp_infos["yaw"].append(yaw)
            kp_infos["roll"].append(roll)

        for k, v in kp_infos.items():
            kp_infos[k] = torch.stack(v)

        return kp_infos
    
    def get_prev_motion(self, x_s_info):
        kp_infos = []
        x_s_info["t"][:, 2] = 0  # zero tz
        if self.motion_generator is not None and self.motion_generator.input_dim == 70:
            x_s_info = self.motion_processer.refine_kp(x_s_info)
            for k, v in x_s_info.items():
                x_s_info[k] = v.reshape(1, -1)

        rescale_ratio = 1.0 if self.motion_mean_std is None else (x_s_info["scale"] + 1e-5) / (self.motion_mean_std["mean"][:, 63:64] + 1e-5)

        for feat_name in ["exp", "scale", "t", "pitch", "yaw", "roll"]:
            if feat_name in ["scale", "t"]:
                # set scale as the mean scale
                kp_infos.append(x_s_info[feat_name] / rescale_ratio)
            else:
                kp_infos.append(x_s_info[feat_name])
        kp_infos = torch.cat(kp_infos, dim=-1)   # B, D
        
        # normalize
        if self.motion_mean_std is not None:
            # normalize exp
            if self.motion_generator is not None and self.motion_generator.input_dim > 70:
                kp_infos[:, :63] = (kp_infos[:, :63] - self.motion_mean_std["mean"][:, :63]) / (self.motion_mean_std["std"][:, :63] + 1e-5)
                # normalize pose
                kp_infos[:, 63:] = kp_infos[:, 63:] - self.motion_mean_std["mean"][:, 63:]
            else:
                kp_infos = (kp_infos - self.motion_mean_std["mean"]) / (self.motion_mean_std["std"] + 1e-5)

        kp_infos = kp_infos.unsqueeze(1)    # B, D
        return kp_infos, rescale_ratio

    def process_audio(self, audio_path: str, silent_audio_path = None, mode="post"):
        logger.debug(f"Processing audio: {audio_path}")
        # add silent audio to pad short input
        ori_audio_path = audio_path
        audio_path, add_frames = self.audio_processor.add_silent_audio(audio_path, silent_audio_path, add_duration=2, linear_fusion=False, mode=mode)
        logger.debug(f"Silent audio padding added: {add_frames} frames")
        audio_emb = self.audio_processor.get_long_audio_emb(audio_path)
        logger.debug(f"Audio embedding extracted, shape: {audio_emb.shape}")
        return audio_emb, audio_path, add_frames, ori_audio_path

    def driven_sample(self, image_path: str, audio_path: str, cfg_scale: float=1., emo: int=8, save_dir=None, smooth=False, silent_audio_path = None, silent_mode="post", return_metrics: bool = False):
        logger.info("=" * 40)
        logger.info("Starting driven_sample generation")
        logger.info("=" * 40)
        
        assert self.motion_generator is not None, f"Motion Generator is not set"
        reference_name = osp.basename(image_path).split('.')[0]
        audio_name = osp.basename(audio_path).split('.')[0]
        
        logger.info(f"Input image: {reference_name}")
        logger.info(f"Input audio: {audio_name}")
        logger.info(f"Parameters: cfg_scale={cfg_scale}, emotion={emo_map.get(emo, 'Unknown')}, smooth={smooth}")
        
        # Initialize timing metrics
        timing_metrics = {
            "total_frames": 0,
            "audio_processing_time": 0.0,
            "motion_generation_time": 0.0,
            "frame_rendering_time": 0.0,
        }
        
        # get audio embeddings
        logger.info("Step 1/5: Processing audio...")
        audio_start = time.time()
        audio_emb, audio_path, add_frames, ori_audio_path = self.process_audio(audio_path, silent_audio_path, mode=silent_mode)
        timing_metrics["audio_processing_time"] = time.time() - audio_start
        logger.debug(f"Audio processing took {timing_metrics['audio_processing_time']:.2f}s")

        # get src image infos
        logger.info("Step 2/5: Processing source image...")
        source_rgb_lst = self.motion_processer.read_image(image_path)
        src_img_256x256, s_lmk, crop_info = self.motion_processer.crop_image(source_rgb_lst[0], do_crop=True)
        f_s, x_s_info = self.motion_processer.prepare_source(src_img_256x256)
        prev_motion, rescale_ratio = self.get_prev_motion(x_s_info)
        
        # generate motions
        logger.info("Step 3/5: Generating motion sequence...")
        motion_start = time.time()
        motion = self.motion_generator.sample(audio_emb, x_s_info["kp"], prev_motion=prev_motion, cfg_scale=cfg_scale, emo=emo)
        if add_frames > 0:
            standard_motion = motion[-max(add_frames*3//4, 1)]
            motion = self.modulate_lip(standard_motion, motion, alpha=5)
            if silent_mode == "both":
                motion = motion[add_frames:-add_frames]
            elif silent_mode == "pre":
                motion = motion[add_frames:]
            else:
                motion = motion[:-add_frames]
        timing_metrics["motion_generation_time"] = time.time() - motion_start
        
        total_frames = len(motion)
        timing_metrics["total_frames"] = total_frames
        motion_fps = total_frames / timing_metrics["motion_generation_time"] if timing_metrics["motion_generation_time"] > 0 else 0
        logger.info(f"Motion sequence generated: {total_frames} frames in {timing_metrics['motion_generation_time']:.2f}s ({motion_fps:.2f} fps)")
        
        kp_infos = self.get_motion_sequence(motion, rescale_ratio=rescale_ratio)
        
        # driven results
        if save_dir is None:
            save_dir = self.cfg.output_dir
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_path = osp.join(save_dir, f'{reference_name}.mp4')
        logger.debug(f"Output path: {save_path}")

        logger.info("Step 4/5: Rendering video frames...")
        render_start = time.time()
        self.motion_processer.driven_by_audio(source_rgb_lst[0], kp_infos, save_path, ori_audio_path, smooth=smooth)
        timing_metrics["frame_rendering_time"] = time.time() - render_start
        
        render_fps = total_frames / timing_metrics["frame_rendering_time"] if timing_metrics["frame_rendering_time"] > 0 else 0
        logger.info(f"Frame rendering completed: {total_frames} frames in {timing_metrics['frame_rendering_time']:.2f}s ({render_fps:.2f} fps)")
        
        logger.info("Step 5/5: Video generation complete")
        logger.info(f"Output saved to: {save_path}")
        
        # Log overall performance summary
        total_time = timing_metrics["audio_processing_time"] + timing_metrics["motion_generation_time"] + timing_metrics["frame_rendering_time"]
        overall_fps = total_frames / total_time if total_time > 0 else 0
        logger.info(f"Performance summary: {total_frames} frames, {total_time:.2f}s total, {overall_fps:.2f} fps overall")
        
        if return_metrics:
            return save_path, timing_metrics
        return save_path



    
    def viz_motion(self, motion_data):
        pass        
        
    def __call__(self):
        pass


if __name__ == "__main__":
    import time
    import random
    import argparse
    
    parser = argparse.ArgumentParser(description="Arguments for the task")
    parser.add_argument('--task', type=str, default="test", help='Task to perform')
    parser.add_argument('--cfg_path', type=str, default="configs/audio2motion/inference/inference.yaml", help='Path to configuration file')
    parser.add_argument('--image_path', type=str, default="src/examples/reference_images/6.jpg", help='Path to the input image')
    parser.add_argument('--audio_path', type=str, default="src/examples/driving_audios/5.wav", help='Path to the driving audio')
    parser.add_argument('--silent_audio_path', type=str, default="src/examples/silent-audio.wav", help='Path to silent audio file')
    parser.add_argument('--save_dir', type=str, default="output/", help='Directory to save results')
    parser.add_argument('--motion_mean_std_path', type=str, default="src/datasets/mean.pt", help='Path to motion mean and standard deviation file')
    parser.add_argument('--cfg_scale', type=float, default=1.2, help='Scaling factor for the configuration')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("MoDA Inference Script")
    logger.info("=" * 60)
    logger.info(f"Image: {args.image_path}")
    logger.info(f"Audio: {args.audio_path}")
    logger.info(f"CFG Scale: {args.cfg_scale}")
    logger.info(f"Output directory: {args.save_dir}")
        
    pipeline = LiveVASAPipeline(cfg_path=args.cfg_path, motion_mean_std_path=args.motion_mean_std_path)
    emo=8
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_dir = osp.join(args.save_dir, f"cfg-{args.cfg_scale}-emo-{emo_map[emo]}")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)  
  
    start_time = time.time()
    video_path = pipeline.driven_sample(
                    args.image_path, args.audio_path, 
                    cfg_scale=args.cfg_scale, emo=emo, 
                    save_dir=save_dir, smooth=False,
                    silent_audio_path = args.silent_audio_path,
                )
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Video generation completed in {elapsed:.2f}s")
    logger.info(f"Output saved to: {video_path}")
    logger.info("=" * 60)


