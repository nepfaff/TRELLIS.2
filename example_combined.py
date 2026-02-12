import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # Can save GPU memory
)
import cv2
import imageio
import numpy as np
from PIL import Image
import torch
import trimesh
from trellis2.pipelines import Trellis2ImageTo3DPipeline, Trellis2TexturingPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 1. Setup Environment Map
# Only used for video rendering. This doesn't affect the GLB file.
envmap = EnvMap(
    torch.tensor(
        cv2.cvtColor(
            cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB,
        ),
        dtype=torch.float32,
        device="cuda",
    )
)

# 2. Load Image-to-3D Pipeline & Run
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
mesh.simplify(16777216)  # nvdiffrast limit

# 3. Render Video
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample.mp4", video, fps=15)

# 4. Export to GLB
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices,
    faces=mesh.faces,
    attr_volume=mesh.attrs,
    coords=mesh.coords,
    attr_layout=mesh.layout,
    voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=1000000,
    texture_size=4096,
    remesh=True,
    remesh_band=1,
    remesh_project=0,
    verbose=True,
)
glb.export("sample.glb", extension_webp=True)

# 5. Free generation pipeline, load texturing pipeline
del pipeline
torch.cuda.empty_cache()

tex_pipeline = Trellis2TexturingPipeline.from_pretrained(
    "microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json"
)
tex_pipeline.cuda()

# 6. Re-texture the GLB mesh with the texturing pipeline
textured = tex_pipeline.run(glb, image)
textured.export("sample_textured.glb", extension_webp=True)
