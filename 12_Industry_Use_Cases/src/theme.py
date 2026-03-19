"""
Shared dark theme for Gradio apps — matches the presentation's
dark blue-green + gold accent design system.

Usage:
    from src.theme import theme, CSS_OVERRIDES
    with gr.Blocks(theme=theme, css=CSS_OVERRIDES) as demo: ...
"""

import gradio as gr

# ── Gold accent color ramp ──
gold_hue = gr.themes.Color(
    c50="#fdf8ed",
    c100="#f9edcc",
    c200="#f3dba0",
    c300="#e9c56d",
    c400="#e2c270",
    c500="#c9a84c",
    c600="#b08f3a",
    c700="#8e722e",
    c800="#6e5823",
    c900="#4e3e19",
    c950="#2e2410",
    name="gold",
)

# ── Dark blue-green neutral ramp ──
dark_neutral = gr.themes.Color(
    c50="#f0f0f0",
    c100="#d4d4d4",
    c200="#a1a1a1",
    c300="#737373",
    c400="#525252",
    c500="#223234",
    c600="#1a2a2e",
    c700="#121e21",
    c800="#0e1618",
    c900="#0a1214",
    c950="#060d0f",
    name="dark_neutral",
)

theme = (
    gr.themes.Base(
        primary_hue=gold_hue,
        neutral_hue=dark_neutral,
        font=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    )
    .set(
        # Body
        body_background_fill="#0a1214",
        body_background_fill_dark="#0a1214",
        body_text_color="#f0f0f0",
        body_text_color_dark="#f0f0f0",
        body_text_color_subdued="#a1a1a1",
        body_text_color_subdued_dark="#a1a1a1",
        # Blocks
        block_background_fill="#121e21",
        block_background_fill_dark="#121e21",
        block_border_color="rgba(255,255,255,0.06)",
        block_border_color_dark="rgba(255,255,255,0.06)",
        block_label_text_color="#a1a1a1",
        block_label_text_color_dark="#a1a1a1",
        block_title_text_color="#f0f0f0",
        block_title_text_color_dark="#f0f0f0",
        # Inputs
        input_background_fill="#1a2a2e",
        input_background_fill_dark="#1a2a2e",
        input_border_color="rgba(255,255,255,0.06)",
        input_border_color_dark="rgba(255,255,255,0.06)",
        input_border_color_focus="rgba(201,168,76,0.18)",
        input_border_color_focus_dark="rgba(201,168,76,0.18)",
        # Buttons primary
        button_primary_background_fill="#c9a84c",
        button_primary_background_fill_dark="#c9a84c",
        button_primary_background_fill_hover="#e2c270",
        button_primary_background_fill_hover_dark="#e2c270",
        button_primary_text_color="#0a1214",
        button_primary_text_color_dark="#0a1214",
        # Buttons secondary
        button_secondary_background_fill="#1a2a2e",
        button_secondary_background_fill_dark="#1a2a2e",
        button_secondary_text_color="#f0f0f0",
        button_secondary_text_color_dark="#f0f0f0",
        # Tables
        table_even_background_fill="#121e21",
        table_even_background_fill_dark="#121e21",
        table_odd_background_fill="#1a2a2e",
        table_odd_background_fill_dark="#1a2a2e",
        # Errors
        error_background_fill="rgba(239,68,68,0.08)",
        error_background_fill_dark="rgba(239,68,68,0.08)",
        error_text_color="#ef4444",
        error_text_color_dark="#ef4444",
        error_border_color="#ef4444",
        error_border_color_dark="#ef4444",
        # Loader / slider
        loader_color="#c9a84c",
        loader_color_dark="#c9a84c",
        slider_color="#c9a84c",
        slider_color_dark="#c9a84c",
        # Shadows — clean flat look
        shadow_drop="none",
        shadow_drop_lg="none",
        shadow_spread="0px",
        shadow_spread_dark="0px",
        # Panel / borders
        panel_background_fill="#121e21",
        panel_background_fill_dark="#121e21",
        panel_border_color="rgba(255,255,255,0.06)",
        panel_border_color_dark="rgba(255,255,255,0.06)",
        border_color_primary="rgba(201,168,76,0.18)",
        border_color_primary_dark="rgba(201,168,76,0.18)",
        # Background fill
        background_fill_primary="#0a1214",
        background_fill_primary_dark="#0a1214",
        background_fill_secondary="#121e21",
        background_fill_secondary_dark="#121e21",
    )
)

CSS_OVERRIDES = """
/* Force dark background on container */
.gradio-container {
    background-color: #0a1214 !important;
    color: #f0f0f0 !important;
}
/* Gold markdown headings */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    color: #c9a84c !important;
}
/* Tab navigation */
.tab-nav button {
    color: #a1a1a1 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected {
    color: #c9a84c !important;
    border-bottom: 2px solid #c9a84c !important;
}
/* Dark scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0a1214; }
::-webkit-scrollbar-thumb { background: #223234; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #2a3f43; }
/* Code blocks */
pre, code {
    background-color: #1a2a2e !important;
}
/* Accordion labels */
.label-wrap span {
    color: #f0f0f0 !important;
}
"""
