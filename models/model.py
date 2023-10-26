import torch
import transformers
from diffusers import DPMSolverMultistepScheduler
from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetPipeline
from src.scripts.qr_generator import BasicQR


class StableDiffusionWithControlNet:
    """
    Stable diffusion model with control net for QR-code generation
    """

    def __init__(
        self,
        device: str = 'cpu',
        brightness: str = "ioclab/control_v1p_sd15_brightness",
        title: str = "lllyasviel/control_v11f1e_sd15_tile",
        stable_diffusion: str = "SG161222/Realistic_Vision_V2.0"
    ):
        # save device
        self.device = device

        # load controlnet models
        self.controlnet_brightness = ControlNetModel.from_pretrained(brightness)
        self.controlnet_tile = ControlNetModel.from_pretrained(title)

        # load stable diffusion
        self.stable_diffusion = StableDiffusionControlNetPipeline.from_pretrained(
            stable_diffusion,
            controlnet=[self.controlnet_brightness, self.controlnet_tile]
            ).to(self.device)

        # define scheduler
        self.stable_diffusion.scheduler = DPMSolverMultistepScheduler.from_config(self.stable_diffusion.scheduler.config, use_karras_sigmas='true')

    def generate(
        self,
        prompt: str,
        qr_text: str,
        width: int = 768,
        height: str = 768,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 30
    ):
        """
        Generates QR-code based on prompt (given to stable diffusion) and qr_text (feeded to basic qr generator)
            > prompt - text that describes image style
            > qr_text - text that should be stored in qr
            > width - width of output image (better bigger for better performance)
            > height - height of output image (better bigger for better performance)
            > num_images_per_prompt - number of output images
            > num_inference_steps - number of inference steps (preferable 30, but 50+ provide better accuracy)
        """

        # define weights and guidance
        controlnets_weights = [0.35, 0.6]
        guidance_starts = [0, 0.3]
        guidance_stops = [1, 0.7]

        # generate target qr image
        qr_img = BasicQR.generate(qr_text)

        # generate images
        results = self.stable_diffusion(
            prompt,
            image=[qr_img, qr_img],
            num_inference_steps=num_inference_steps,
            width=width, height=height,
            num_images_per_prompt=num_images_per_prompt,
            control_guidance_start=guidance_starts,
            control_guidance_end=guidance_stops,
            controlnet_conditioning_scale=controlnets_weights
        )

        return results.images[0]

