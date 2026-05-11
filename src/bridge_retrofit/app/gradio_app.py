"""Gradio application.

This module is optional; it only runs when gradio is installed.
"""

from __future__ import annotations

from typing import Any

from bridge_retrofit.config import ProjectConfig
from bridge_retrofit.pipeline import run_predict
from bridge_retrofit.preprocessing.pipeline import coerce_jsonable


def launch(cfg: ProjectConfig, share: bool = False) -> None:
    try:
        import gradio as gr
    except Exception as e:  # pragma: no cover
        raise RuntimeError("gradio is not installed; install requirements-colab.txt") from e

    def infer(payload_text: str) -> dict[str, Any]:
        import json

        payload = json.loads(payload_text)
        out = run_predict(cfg, payload)
        return coerce_jsonable(out)

    demo = gr.Interface(
        fn=infer,
        inputs=gr.Textbox(label="Input JSON", lines=8, placeholder='{"Bridge_Type": "Beam", "Age_Years": 25}'),
        outputs=gr.JSON(label="Output"),
        title="Bridge Failure Analysis (Demo)",
    )

    demo.launch(share=share)
