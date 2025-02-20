import argparse
import os
import pickle
import threading

import torch
from flask import Blueprint, Flask, Response, request
from flask_restful import Api, Resource

device = f'cuda:{torch.cuda.device_count()-1}'
dtype = torch.bfloat16


def parsed_args():
    parser = argparse.ArgumentParser(description="StepVideo API Functions")
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--clip_dir', type=str, default='hunyuan_clip')
    parser.add_argument('--llm_dir', type=str, default='step_llm')
    parser.add_argument('--vae_dir', type=str, default='vae')
    parser.add_argument('--port', type=str, default='8080')
    args = parser.parse_args()
    return args


class StepVaePipeline(Resource):

    def __init__(self, vae_dir, version=2):
        self.vae = self.build_vae(vae_dir, version)
        self.scale_factor = 1.0

    def build_vae(self, vae_dir, version=2):
        from fastvideo.models.stepvideo.vae.vae import AutoencoderKL
        (model_name, z_channels) = ("vae_v2.safetensors", 64) if version == 2 else ("vae.safetensors", 16)
        model_path = os.path.join(vae_dir, model_name)

        model = AutoencoderKL(
            z_channels=z_channels,
            model_path=model_path,
            version=version,
        ).to(dtype).to(device).eval()
        print("Initialized vae...")
        return model

    def decode(self, samples, *args, **kwargs):
        with torch.no_grad():
            try:
                dtype = next(self.vae.parameters()).dtype
                device = next(self.vae.parameters()).device
                samples = self.vae.decode(samples.to(dtype).to(device) / self.scale_factor)
                if hasattr(samples, 'sample'):
                    samples = samples.sample
                return samples
            except:
                torch.cuda.empty_cache()
                return None


lock = threading.Lock()


class VAEapi(Resource):

    def __init__(self, vae_pipeline):
        self.vae_pipeline = vae_pipeline

    def get(self):
        with lock:
            try:
                feature = pickle.loads(request.get_data())
                feature['api'] = 'vae'

                feature = {k: v for k, v in feature.items() if v is not None}
                video_latents = self.vae_pipeline.decode(**feature)
                response = pickle.dumps(video_latents)

            except Exception as e:
                print("Caught Exception: ", e)
                return Response(e)

            return Response(response)


class CaptionPipeline(Resource):

    def __init__(self, llm_dir, clip_dir):
        self.text_encoder = self.build_llm(llm_dir)
        self.clip = self.build_clip(clip_dir)

    def build_llm(self, model_dir):
        from fastvideo.models.stepvideo.text_encoder.stepllm import STEP1TextEncoder
        text_encoder = STEP1TextEncoder(model_dir, max_length=320).to(dtype).to(device).eval()
        print("Initialized text encoder...")
        return text_encoder

    def build_clip(self, model_dir):
        from fastvideo.models.stepvideo.text_encoder.clip import HunyuanClip
        clip = HunyuanClip(model_dir, max_length=77).to(device).eval()
        print("Initialized clip encoder...")
        return clip

    def embedding(self, prompts, *args, **kwargs):
        with torch.no_grad():
            try:
                y, y_mask = self.text_encoder(prompts)

                clip_embedding, _ = self.clip(prompts)

                len_clip = clip_embedding.shape[1]
                y_mask = torch.nn.functional.pad(y_mask, (len_clip, 0),
                                                 value=1)  ## pad attention_mask with clip's length

                data = {
                    'y': y.detach().cpu(),
                    'y_mask': y_mask.detach().cpu(),
                    'clip_embedding': clip_embedding.to(torch.bfloat16).detach().cpu()
                }

                return data
            except Exception as err:
                print(f"{err}")
                return None


lock = threading.Lock()


class Captionapi(Resource):

    def __init__(self, caption_pipeline):
        self.caption_pipeline = caption_pipeline

    def get(self):
        with lock:
            try:
                feature = pickle.loads(request.get_data())
                feature['api'] = 'caption'

                feature = {k: v for k, v in feature.items() if v is not None}
                embeddings = self.caption_pipeline.embedding(**feature)
                response = pickle.dumps(embeddings)

            except Exception as e:
                print("Caught Exception: ", e)
                return Response(e)

            return Response(response)


class RemoteServer(object):

    def __init__(self, args) -> None:
        self.app = Flask(__name__)
        root = Blueprint("root", __name__)
        self.app.register_blueprint(root)
        api = Api(self.app)

        self.vae_pipeline = StepVaePipeline(vae_dir=os.path.join(args.model_dir, args.vae_dir))
        api.add_resource(
            VAEapi,
            "/vae-api",
            resource_class_args=[self.vae_pipeline],
        )

        self.caption_pipeline = CaptionPipeline(llm_dir=os.path.join(args.model_dir, args.llm_dir),
                                                clip_dir=os.path.join(args.model_dir, args.clip_dir))
        api.add_resource(
            Captionapi,
            "/caption-api",
            resource_class_args=[self.caption_pipeline],
        )

    def run(self, host="0.0.0.0", port=8080):
        self.app.run(host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    args = parsed_args()
    flask_server = RemoteServer(args)
    flask_server.run(host="0.0.0.0", port=args.port)
