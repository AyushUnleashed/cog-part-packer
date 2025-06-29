import os
import sys
import tempfile
from datetime import datetime
from typing import Optional

import cv2
import kiui
import numpy as np
import rembg
import torch
import torch.nn as nn
import trimesh
from cog import BasePredictor, Input, Path

# Add PartPacker to Python path so its internal imports work
partpacker_path = os.path.join(os.path.dirname(__file__), "partpacker")
if partpacker_path not in sys.path:
    sys.path.insert(0, partpacker_path)

# Now import the PartPacker components
from flow.configs.schema import ModelConfig
from flow.model import Model
from flow.utils import get_random_color, recenter_foreground
from vae.utils import postprocess_mesh

# Import our utility functions and download function
from download_weights import download_weights, verify_downloads

# Memory optimization setting
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Constants
TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)
MAX_SEED = np.iinfo(np.int32).max

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Setting up PartPacker model...")
        
        # Check GPU memory first
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            print(f"GPU memory free: {torch.cuda.memory_reserved(0) / 1024**3:.1f}GB")
        else:
            raise RuntimeError("CUDA not available!")
        
        # Download weights if not present
        if not verify_downloads():
            print("Downloading model weights...")
            download_weights()
        
        print("Weights verified, initializing background remover...")
        # Initialize background remover
        self.bg_remover = rembg.new_session()
        
        print("Loading checkpoint...")
        # Load checkpoint FIRST (following original infer.py pattern)
        ckpt_dict = torch.load("pretrained/flow.pt", weights_only=True)
        
        # Extract model weights if nested
        if "model" in ckpt_dict:
            ckpt_dict = ckpt_dict["model"]
        
        print("Creating model configuration...")
        # Model configuration
        model_config = ModelConfig(
            vae_conf="vae.configs.part_woenc",
            vae_ckpt_path="pretrained/vae.pt",
            qknorm=True,
            qknorm_type="RMSNorm",
            use_pos_embed=False,
            dino_model="dinov2_vitg14",
            hidden_dim=1536,
            flow_shift=3.0,
            logitnorm_mean=1.0,
            logitnorm_std=1.0,
            latent_size=4096,
            use_parts=True,
        )
        
        print("Initializing model...")
        # Initialize model AFTER loading checkpoint
        self.model = Model(model_config).eval().cuda().bfloat16()
        
        print("Loading weights into model...")
        # Load weights
        try:
            self.model.load_state_dict(ckpt_dict, strict=True)
            print("PartPacker model loaded successfully!")
            
            # Check memory after loading
            print(f"GPU memory after loading: {torch.cuda.memory_allocated(0) / 1024**3:.1f}GB allocated")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.1f}GB")
            
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise e

    def process_image(self, image_path: str) -> np.ndarray:
        """Process input image for the model - matches original infer.py approach"""
        try:
            # Try to read image (closest to kiui.read_image behavior)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert to RGBA format (matching kiui.read_image order="RGBA")
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            elif image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Add alpha channel
                alpha = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
                image = np.concatenate([image, alpha], axis=2)
            
            # bg removal if there is no alpha channel or if it's all opaque
            if image.shape[-1] == 3 or np.all(image[..., -1] == 255):
                image = rembg.remove(image, session=self.bg_remover)  # [H, W, 4]
            
            mask = image[..., -1] > 0
            if not mask.any():
                print("Warning: No foreground object detected after background removal")
            
            # Match original preprocessing exactly
            image = recenter_foreground(image, mask, border_ratio=0.1)
            image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_LINEAR)  # Use LINEAR like original
            image = image.astype(np.float32) / 255.0
            image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])  # white background
            return image
            
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

    def predict(
        self,
        image: Path = Input(
            description="Input image for 3D object generation"
        ),
        num_steps: int = Input(
            description="Number of inference steps (higher = better quality, slower)",
            default=50,
            ge=1,
            le=100
        ),
        cfg_scale: float = Input(
            description="Classifier-free guidance scale",
            default=7.0,
            ge=1.0,
            le=20.0
        ),
        grid_resolution: int = Input(
            description="Grid resolution for mesh extraction",
            default=384,
            ge=256,
            le=512
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible results (leave blank for random)",
            default=None
        ),
        simplify_mesh: bool = Input(
            description="Simplify the output mesh",
            default=False
        ),
        target_num_faces: int = Input(
            description="Target number of faces for mesh simplification",
            default=100000,
            ge=10000,
            le=1000000
        ),
    ) -> Path:
        """Generate 3D object from input image"""
        
        # Set seed
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        kiui.seed_everything(seed)
        
        # Process input image
        print("Processing input image...")
        processed_image = self.process_image(str(image))
        
        # Convert to tensor (matching original infer.py exactly)
        image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).contiguous().unsqueeze(0).float().cuda()
        
        print(f"🎬 Generating 3D object with {num_steps} steps at {grid_resolution} resolution")
        print(f"⚙️ CFG scale: {cfg_scale}, Seed: {seed}")
        
        # Prepare data for model
        data = {"cond_images": image_tensor}
        
        # Run inference
        print("Running flow model inference...")
        try:
            with torch.inference_mode():
                results = self.model(data, num_steps=num_steps, cfg_scale=cfg_scale)
        except Exception as e:
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                raise RuntimeError(
                    f"GPU memory error. Try reducing grid_resolution (current: {grid_resolution}) "
                    f"or num_steps (current: {num_steps})"
                )
            raise RuntimeError(f"Error during flow model inference: {e}")
        
        latent = results["latent"]
        
        # Generate meshes for both parts
        print("Extracting meshes from latent representation...")
        data_part0 = {"latent": latent[:, : self.model.config.latent_size, :]}
        data_part1 = {"latent": latent[:, self.model.config.latent_size :, :]}
        
        try:
            with torch.inference_mode():
                results_part0 = self.model.vae(data_part0, resolution=grid_resolution)
                results_part1 = self.model.vae(data_part1, resolution=grid_resolution)
        except Exception as e:
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                raise RuntimeError(
                    f"GPU memory error during mesh extraction. Try reducing grid_resolution (current: {grid_resolution})"
                )
            raise RuntimeError(f"Error during mesh extraction: {e}")
        
        # Process meshes
        if not simplify_mesh:
            target_num_faces = -1
        
        # Part 0
        vertices, faces = results_part0["meshes"][0]
        mesh_part0 = trimesh.Trimesh(vertices, faces)
        mesh_part0.vertices = mesh_part0.vertices @ TRIMESH_GLB_EXPORT.T
        mesh_part0 = postprocess_mesh(mesh_part0, target_num_faces)
        parts = mesh_part0.split(only_watertight=False)
        
        # Part 1
        vertices, faces = results_part1["meshes"][0]
        mesh_part1 = trimesh.Trimesh(vertices, faces)
        mesh_part1.vertices = mesh_part1.vertices @ TRIMESH_GLB_EXPORT.T
        mesh_part1 = postprocess_mesh(mesh_part1, target_num_faces)
        parts.extend(mesh_part1.split(only_watertight=False))
        
        # Filter out parts with too few faces
        parts = [part for part in parts if len(part.faces) > 10]
        
        if not parts:
            raise RuntimeError("No valid mesh parts generated. Try different parameters or input image.")
        
        print(f"Generated {len(parts)} valid mesh parts")
        
        # Assign different colors to each part
        for j, part in enumerate(parts):
            part.visual.vertex_colors = get_random_color(j, use_float=True)
        
        # Create scene and export
        mesh_scene = trimesh.Scene(parts)
        
        # Save output
        output_path = Path(tempfile.mkdtemp()) / "output.glb"
        try:
            mesh_scene.export(str(output_path))
        except Exception as e:
            raise RuntimeError(f"Error saving mesh: {e}")
        
        print(f"✅ Generated 3D object with {len(parts)} parts")
        return output_path