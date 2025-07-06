"""
PartPacker Cog Predictor
Efficient Part-level 3D Object Generation via Dual Volume Packing
"""

import importlib
import os
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, BaseModel, Path as CogPath
from PIL import Image

# Add PartPacker submodule to Python path
PARTPACKER_PATH = os.path.join(os.path.dirname(__file__), "partpacker")
sys.path.insert(0, PARTPACKER_PATH)

# Import PartPacker modules after adding to path
import kiui
import rembg
import trimesh
from flow.model import Model
from flow.utils import get_random_color, recenter_foreground
from vae.utils import postprocess_mesh


class PredictOutput(BaseModel):
    output_zip_path: CogPath | None = None
    combined_model_path: CogPath | None = None



class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Transformation matrix for GLB export
        self.TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)

        # Initialize background remover
        self.bg_remover = rembg.new_session()
        
        # Load weights
        ckpt_dict = torch.load("pretrained/flow.pt", weights_only=True)

        # delete all keys other than model
        if "model" in ckpt_dict:
            ckpt_dict = ckpt_dict["model"]
        
        self.config_file_path = "flow.configs.big_parts_strict_pvae"
        model_config = importlib.import_module(self.config_file_path).make_config()
        
        # Load model
        print("Loading PartPacker model...")
        self.model = Model(model_config).eval().cuda().bfloat16()
        

        self.model.load_state_dict(ckpt_dict, strict=True)
        print("Model loaded successfully!")


    def preprocess_image(self, path):
        input_image = kiui.read_image(path, mode="uint8", order="RGBA")

        # bg removal if there is no alpha channel
        if input_image.shape[-1] == 3:
            input_image = rembg.remove(input_image, session=self.bg_remover)  # [H, W, 4]

        mask = input_image[..., -1] > 0
        image = recenter_foreground(input_image, mask, border_ratio=0.1)
        image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])  # white background
        return image


    def predict(
        self,
        image: CogPath = Input(description="Input image for 3D object generation"),
        num_steps: int = Input(
            description="Number of inference steps", 
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
            description="Random seed for reproducible results", 
            default=None
        ),
        simplify_mesh: bool = Input(
            description="Whether to simplify the output mesh", 
            default=False
        ),
        target_num_faces: int = Input(
            description="Target number of faces for mesh simplification", 
            default=100000, 
            ge=10000, 
            le=1000000
        ),
    ) -> PredictOutput:
        """Generate 3D object from input image"""
        
        # Set random seed
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        kiui.seed_everything(seed)
        
        # Create temporary working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Preprocess image
            print("Preprocessing image...")
            input_image = self.preprocess_image(str(image))
            
            # Save processed image
            processed_image_path = temp_path / "processed_input.jpg"
            kiui.write_image(processed_image_path,input_image)
            
            # Prepare image tensor
            image_tensor = torch.from_numpy(input_image).permute(2, 0, 1).contiguous().unsqueeze(0).float().cuda()
            
            # Run inference
            print(f"Running inference with {num_steps} steps, CFG scale {cfg_scale}...")
            data = {"cond_images": image_tensor}
            
            with torch.inference_mode():
                results = self.model(data, num_steps=num_steps, cfg_scale=cfg_scale)
            
            latent = results["latent"]
            # kiui.lo(latent)

            # Extract meshes for both parts
            print("Extracting meshes...")
            data_part0 = {"latent": latent[:, :self.model.config.latent_size, :]}
            data_part1 = {"latent": latent[:, self.model.config.latent_size:, :]}
            
            with torch.inference_mode():
                results_part0 = self.model.vae(data_part0, resolution=grid_resolution)
                results_part1 = self.model.vae(data_part1, resolution=grid_resolution)
            
            # Process part 0
            vertices, faces = results_part0["meshes"][0]
            mesh_part0 = trimesh.Trimesh(vertices, faces)
            mesh_part0.vertices = mesh_part0.vertices @ self.TRIMESH_GLB_EXPORT.T
            
            if simplify_mesh:
                mesh_part0 = postprocess_mesh(mesh_part0, target_num_faces)
            else:
                mesh_part0 = postprocess_mesh(mesh_part0, -1)
                
            parts = mesh_part0.split(only_watertight=False)
            
            # Process part 1
            vertices, faces = results_part1["meshes"][0]
            mesh_part1 = trimesh.Trimesh(vertices, faces)
            mesh_part1.vertices = mesh_part1.vertices @ self.TRIMESH_GLB_EXPORT.T
            
            if simplify_mesh:
                mesh_part1 = postprocess_mesh(mesh_part1, target_num_faces)
            else:
                mesh_part1 = postprocess_mesh(mesh_part1, -1)
                
            parts.extend(mesh_part1.split(only_watertight=False))
            
            # Filter out tiny parts (< 10 faces)
            parts = [part for part in parts if len(part.faces) > 10]
            
            print(f"Generated {len(parts)} parts")
            
            # Assign random colors to parts
            for j, part in enumerate(parts):
                part.visual.vertex_colors = get_random_color(j, use_float=True)
            
            # Export files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"partpacker_{timestamp}"
            
            # Export combined mesh
            combined_scene = trimesh.Scene(parts)
            combined_path = temp_path / f"{base_name}_combined.glb"
            combined_scene.export(str(combined_path))
            
            # Export individual parts
            part_paths = []
            for j, part in enumerate(parts):
                part_path = temp_path / f"{base_name}_part_{j:02d}.glb"
                part.export(str(part_path))
                part_paths.append(part_path)
            
            # Export dual volumes
            vol0_path = temp_path / f"{base_name}_volume_0.glb"
            vol1_path = temp_path / f"{base_name}_volume_1.glb"
            mesh_part0.export(str(vol0_path))
            mesh_part1.export(str(vol1_path))
            
            # Create zip file with all outputs
            output_zip_path = temp_path / f"{base_name}_output.zip"
            
            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add processed input image
                zipf.write(processed_image_path, "processed_input.jpg")
                
                # Add combined mesh
                zipf.write(combined_path, f"{base_name}_combined.glb")
                
                # Add dual volumes
                zipf.write(vol0_path, f"{base_name}_volume_0.glb")
                zipf.write(vol1_path, f"{base_name}_volume_1.glb")
                
                # Add individual parts
                for j, part_path in enumerate(part_paths):
                    zipf.write(part_path, f"parts/{base_name}_part_{j:02d}.glb")
                
                # Add generation info
                info_content = f"""
                PartPacker Generation Info
                ========================
                Timestamp: {timestamp}
                Seed: {seed}
                Inference Steps: {num_steps}
                CFG Scale: {cfg_scale}
                Grid Resolution: {grid_resolution}
                Mesh Simplified: {simplify_mesh}
                Target Faces: {target_num_faces if simplify_mesh else 'No limit'}
                Total Parts Generated: {len(parts)}

                Files Included:
                - processed_input.jpg: Preprocessed input image
                - {base_name}_combined.glb: All parts combined with random colors
                - {base_name}_volume_0.glb: First dual volume
                - {base_name}_volume_1.glb: Second dual volume  
                - parts/: Individual part files

                Usage:
                The combined GLB file contains all parts and can be imported into Blender, Unity, etc.
                Each part can be separated for individual manipulation.
                The dual volumes show the two main components used in the generation process.
                """
                zipf.writestr("generation_info.txt", info_content)
            
            print(f"Generation complete! Created {len(parts)} parts")
            
            # Copy zip to output location
            final_output_path = f"/tmp/{base_name}_output.zip"
            final_combined_path = f"/tmp/{base_name}_combined.glb"
            os.rename(str(output_zip_path), final_output_path)
            os.rename(str(combined_path), final_combined_path)

            
            return PredictOutput(
                output_zip_path=CogPath(final_output_path),
                combined_model_path=CogPath(final_combined_path)
            )