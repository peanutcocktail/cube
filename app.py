import os
import torch
import gradio as gr
import trimesh
import platform
from cube3d.inference.engine import Engine, EngineFast
from cube3d.mesh_utils.postprocessing import (
    PYMESHLAB_AVAILABLE,
    create_pymeshset,
    postprocess_mesh,
    save_mesh,
)


def generate_mesh(
    prompt,
    fast_inference,
    resolution_base,
    disable_postprocessing,
    output_dir="outputs"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine available device (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Default paths
    config_path = "cube3d/configs/open_model.yaml"
    gpt_ckpt_path = "model_weights/shape_gpt.safetensors"
    shape_ckpt_path = "model_weights/shape_tokenizer.safetensors"
    
    # Initialize engine based on device and fast_inference flag
    # Note: EngineFast is only compatible with CUDA
    if fast_inference and device.type == "cuda":
        engine = EngineFast(
            config_path, gpt_ckpt_path, shape_ckpt_path, device=device
        )
    else:
        engine = Engine(
            config_path, gpt_ckpt_path, shape_ckpt_path, device=device
        )
    
    # Generate mesh
    mesh_v_f = engine.t2s([prompt], use_kv_cache=True, resolution_base=resolution_base)
    vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
    output_name = "output"
    obj_path = os.path.join(output_dir, f"{output_name}.obj")
    
    if PYMESHLAB_AVAILABLE:
        ms = create_pymeshset(vertices, faces)
        if not disable_postprocessing:
            target_face_num = max(10000, int(faces.shape[0] * 0.1))
            postprocess_mesh(ms, target_face_num, obj_path)
        save_mesh(ms, obj_path)
    else:
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(obj_path)
    
    # Return both the file path for download and the same path for 3D display
    return [obj_path, obj_path]

def app():
    # Determine device for display
    if torch.cuda.is_available():
        device_info = f"CUDA: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_info = "Apple Silicon (MPS)"
    else:
        device_info = "CPU"
    
    with gr.Blocks(title="Cube3D - Text to 3D Shape Generator") as demo:
        gr.Markdown("# Cube3D - Text to 3D Shape Generator")
        gr.Markdown("Generate 3D models from text prompts using Roblox's Cube3D model")
        gr.Markdown(f"**Running on:** {device_info}")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Text Prompt", 
                    placeholder="Broad-winged flying red dragon, elongated, folded legs.",
                    lines=3
                )
                
                with gr.Accordion("Model Settings", open=False):
                    # Only enable fast_inference checkbox if CUDA is available
                    cuda_available = torch.cuda.is_available()
                    fast_inference = gr.Checkbox(
                        label="Fast Inference (CUDA only)", 
                        value=cuda_available,
                        interactive=cuda_available,
                        info="Only available on NVIDIA GPUs with CUDA"
                    )
                    resolution_base = gr.Slider(
                        label="Resolution Base", 
                        minimum=4.0, 
                        maximum=9.0, 
                        step=0.5, 
                        value=8.0
                    )
                    disable_postprocessing = gr.Checkbox(
                        label="Disable Mesh Postprocessing", 
                        value=False
                    )
                
                generate_btn = gr.Button("Generate 3D Model", variant="primary")
                obj_file = gr.File(label="Download OBJ File")
            
            with gr.Column(scale=2):
                model_3d = gr.Model3D(
                    label="3D Model Preview",
                    clear_color=[0.0, 0.0, 0.0, 0.0],  # Transparent background
                )
        
        generate_btn.click(
            fn=generate_mesh,
            inputs=[
                prompt, 
                fast_inference, 
                resolution_base,
                disable_postprocessing
            ],
            outputs=[obj_file, model_3d]
        )
        
        gr.Markdown("""
        ## Instructions
        1. Enter a text prompt describing the 3D shape you want to generate
        2. Adjust model settings if needed
        3. Click "Generate 3D Model" button
        4. Download the OBJ file or view the turntable animation if enabled
        
        **Notes:** 
        - First-time generation may take longer as the model is being loaded
        - The app supports CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU backends
        - Fast inference is only available on CUDA devices
        - For Apple Silicon, the generation process will automatically use the MPS backend for improved performance
        """)
    
    return demo

if __name__ == "__main__":
    demo = app()
    demo.launch()
