"""
Gradio-based local client for TRELLIS 3D generation.

Provides a user interface for:
- Image upload and preprocessing
- Generation parameter controls
- Real-time progress display
- 3D model preview and download
"""

from __future__ import annotations

import base64
import os
import tempfile

import gradio as gr

from .api import APIError, TrellisAPIClient


def get_client() -> TrellisAPIClient | None:
    """
    Create API client from environment variables.

    Requires TRELLIS_API_URL and TRELLIS_API_KEY to be set.

    Returns:
        TrellisAPIClient if configured, None otherwise
    """
    api_url = os.environ.get("TRELLIS_API_URL")
    api_key = os.environ.get("TRELLIS_API_KEY")

    if not api_url or not api_key:
        return None

    return TrellisAPIClient(base_url=api_url, api_key=api_key)


def generate_3d(
    image: str | None,
    seed: int,
    ss_sampling_steps: int,
    slat_sampling_steps: int,
    slat_guidance_strength: float,
) -> tuple[str | None, str | None, str]:
    """
    Generate 3D from uploaded image.

    Args:
        image: Path to uploaded image (from gr.Image)
        seed: Random seed
        ss_sampling_steps: Sparse structure steps
        slat_sampling_steps: SLAT steps
        slat_guidance_strength: Guidance strength

    Returns:
        Tuple of (video_path, state_json, status_message)
    """
    if image is None:
        return None, None, "Please upload an image first."

    client = get_client()
    if client is None:
        return None, None, "Error: TRELLIS_API_URL and TRELLIS_API_KEY not set."

    try:
        # Call generate endpoint
        result = client.generate(
            image_path=image,
            seed=seed,
            ss_sampling_steps=ss_sampling_steps,
            slat_sampling_steps=slat_sampling_steps,
            slat_guidance_strength=slat_guidance_strength,
        )

        # Decode video and save to temp file
        video_bytes = base64.b64decode(result["video"])
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            video_path = f.name

        # Build status message
        elapsed = client.last_request_elapsed or 0
        cold_start_msg = " (cold start detected)" if client.was_cold_start() else ""
        status = f"Generated in {elapsed:.1f}s{cold_start_msg}"

        # Return state as JSON string for gr.State
        return video_path, result["state"], status

    except APIError as e:
        return None, None, f"API Error: {e.code} - {e.message}"
    except Exception as e:
        return None, None, f"Error: {e}"


def extract_model(
    state: str | None,
    output_format: str,
    mesh_simplify_ratio: float,
    texture_size: int,
) -> tuple[str | None, str]:
    """
    Extract 3D model from generation state.

    Args:
        state: Base64 compressed state from generate()
        output_format: "GLB" or "Gaussian (PLY)"
        mesh_simplify_ratio: Mesh simplification (GLB only)
        texture_size: Texture resolution (GLB only)

    Returns:
        Tuple of (model_path, status_message)
    """
    if state is None:
        return None, "Please generate a 3D model first."

    client = get_client()
    if client is None:
        return None, "Error: TRELLIS_API_URL and TRELLIS_API_KEY not set."

    try:
        # Create temp file for output
        suffix = ".glb" if output_format == "GLB" else ".ply"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            output_path = f.name

        # Call appropriate extract endpoint
        if output_format == "GLB":
            client.extract_glb(
                state=state,
                mesh_simplify_ratio=mesh_simplify_ratio,
                texture_size=texture_size,
                output_path=output_path,
            )
        else:
            client.extract_gaussian(
                state=state,
                output_path=output_path,
            )

        elapsed = client.last_request_elapsed or 0
        return output_path, f"Extracted in {elapsed:.1f}s"

    except APIError as e:
        return None, f"API Error: {e.code} - {e.message}"
    except Exception as e:
        return None, f"Error: {e}"


def create_interface() -> gr.Blocks:
    """
    Create and configure the Gradio interface.

    Returns:
        Configured Gradio Blocks interface
    """
    with gr.Blocks(title="TRELLIS 3D Generator") as demo:
        gr.Markdown("# TRELLIS 3D Generator")
        gr.Markdown("Generate 3D models from images using the Modal TRELLIS service.")

        # State for storing generation result
        generation_state = gr.State(value=None)

        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Input Image",
                    type="filepath",
                    height=300,
                )

                with gr.Accordion("Generation Parameters", open=True):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=999999,
                        value=42,
                        step=1,
                    )
                    ss_steps = gr.Slider(
                        label="Sparse Structure Steps",
                        minimum=1,
                        maximum=50,
                        value=12,
                        step=1,
                    )
                    slat_steps = gr.Slider(
                        label="SLAT Steps",
                        minimum=1,
                        maximum=50,
                        value=12,
                        step=1,
                    )
                    guidance = gr.Slider(
                        label="Guidance Strength",
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.1,
                    )

                generate_btn = gr.Button("Generate 3D", variant="primary")

            # Right column: Output
            with gr.Column(scale=1):
                video_output = gr.Video(
                    label="Preview Video",
                    height=300,
                )
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

        # Extraction section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Extract Model")
                with gr.Row():
                    output_format = gr.Radio(
                        label="Output Format",
                        choices=["GLB", "Gaussian (PLY)"],
                        value="GLB",
                    )
                with gr.Row():
                    mesh_simplify = gr.Slider(
                        label="Mesh Simplify Ratio",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                    )
                    texture_size = gr.Dropdown(
                        label="Texture Size",
                        choices=[512, 1024, 2048],
                        value=1024,
                    )
                extract_btn = gr.Button("Extract Model")

            with gr.Column():
                model_output = gr.File(
                    label="Download Model",
                )
                extract_status = gr.Textbox(
                    label="Extraction Status",
                    interactive=False,
                )

        # Wire up events
        generate_btn.click(
            fn=generate_3d,
            inputs=[image_input, seed, ss_steps, slat_steps, guidance],
            outputs=[video_output, generation_state, status_output],
        )

        extract_btn.click(
            fn=extract_model,
            inputs=[generation_state, output_format, mesh_simplify, texture_size],
            outputs=[model_output, extract_status],
        )

    return demo


def main() -> None:
    """
    Entry point for the client application.

    Loads configuration, creates the interface, and launches the server.
    """
    # Check for required environment variables
    api_url = os.environ.get("TRELLIS_API_URL")
    api_key = os.environ.get("TRELLIS_API_KEY")

    if not api_url or not api_key:
        print("Warning: TRELLIS_API_URL and TRELLIS_API_KEY not set.")
        print("Set these environment variables to connect to the Modal service.")
        print()

    # Create and launch interface
    demo = create_interface()
    demo.launch()


if __name__ == "__main__":
    main()
