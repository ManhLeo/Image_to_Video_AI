"""Gradio entrypoint for AI Photo Studio — Premium Light Theme v4 (English)."""

import gradio as gr
from services.batch_processor import BatchProcessor
from services.video_generator import VideoGenerator, VideoConfig
from utils.file_utils import get_output_filename, clean_temp_dir
from utils.system_utils import format_vram_status
import config


# ───────────────────────────── State helpers ─────────────────────────────
initial_state = {
    "uploaded_paths": [],
    "user_config": {},
    "analysis_results": [],
    "selected_paths": [],
    "video_path": None,
}


def _field(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _result_to_payload(result):
    return {
        "image_path": str(_field(result, "image_path", "")),
        "filename": str(_field(result, "filename", "")),
        "overall_score": float(_field(result, "overall_score", 0.0)),
        "smile_score": float(_field(result, "smile_score", 0.0)),
        "eye_open_score": float(_field(result, "eye_open_score", 0.0)),
        "sharpness_score": float(_field(result, "sharpness_score", 0.0)),
        "exposure_score": float(_field(result, "exposure_score", 0.0)),
        "aesthetic_score": float(_field(result, "aesthetic_score", 0.0)),
        "score_label": str(_field(result, "score_label", "")),
        "brief_note": str(_field(result, "brief_note", "")),
        "auto_selected": bool(_field(result, "auto_selected", False)),
        "analysis_result": _field(result, "analysis_result", {}),
    }


# ───────────────────────────── Event handlers ─────────────────────────────
def on_upload(files):
    if not files:
        return [], 0, []
    paths = [f.name for f in files]
    return paths, len(paths), paths


def on_analyze(uploaded_paths, theme, priorities, sensitivity, progress=gr.Progress()):
    if not uploaded_paths:
        raise gr.Error("No images uploaded!")

    theme_map = {"family": "family", "individual": "individual", "group": "group"}
    priority_map = {
        "smile": "smile", "quality": "quality",
        "lighting": "lighting", "aesthetic": "aesthetic",
    }

    try:
        normalized_sensitivity = max(5, min(10, int(sensitivity)))
    except (TypeError, ValueError):
        normalized_sensitivity = 7

    user_config = {
        "theme": theme_map.get(theme.split("/")[0].strip().lower(), "family"),
        "priorities": [priority_map.get(p.split("/")[0].strip().lower(), p) for p in priorities],
        "sensitivity": normalized_sensitivity,
    }

    processor = BatchProcessor()

    def update_progress(current, total, latest_result, progress_pct):
        if latest_result:
            desc = f"Analyzing photo {current}/{total}: {latest_result.filename} — {latest_result.score_label}"
        else:
            desc = f"Processing... {current}/{total}"
        progress(progress_pct, desc=desc)

    results = processor.process_all(uploaded_paths, user_config, progress_callback=update_progress)
    summary = processor.get_summary()
    processor.close()

    payload_results = [_result_to_payload(r) for r in results]

    gallery_data = [
        (str(r["image_path"]).replace("\\", "/"), f"{r['filename']} — {int(r['overall_score'])} pts")
        for r in payload_results
    ]

    df_rows = [
        [r["filename"], round(r["overall_score"], 1), round(r["smile_score"], 1),
         round(r["eye_open_score"], 1), round(r["sharpness_score"], 1),
         round(r["exposure_score"], 1), round(r["aesthetic_score"], 1),
         r["score_label"], r["brief_note"]]
        for r in payload_results
    ]

    all_filenames = list(dict.fromkeys([r["filename"] for r in payload_results]))
    selected_filenames = [r["filename"] for r in payload_results if r["auto_selected"]]
    selected_paths = [r["image_path"].replace("\\", "/") for r in payload_results if r["auto_selected"]]

    return (
        payload_results, selected_paths, gallery_data, df_rows,
        gr.update(choices=all_filenames, value=selected_filenames),
        int(summary.get("selected_count", 0)),
        int(summary.get("rejected_count", 0)),
        round(summary.get("avg_score", 0), 1),
        round(summary.get("avg_smile", 0), 1),
        f"✔ Successfully processed {len(results)} images.",
        format_vram_status(),
        "✅ Analysis complete.",
    )


def on_selection_change_combined(selected_filenames, analysis_results, slide_duration):
    if not selected_filenames:
        return [], "No photos selected!"
    selected_paths = []
    for filename in selected_filenames:
        for r in analysis_results:
            if _field(r, "filename", "") == filename:
                selected_paths.append(_field(r, "image_path", ""))
    count = len(selected_paths)
    total_time = count * slide_duration
    info_msg = (
        f"**Video Config:** {count} photos · Estimated ~{total_time:.1f} seconds"
        if count > 0 else "No photos selected!"
    )
    return selected_paths, info_msg


def update_analysis_display(results, min_score, sort_by, show_type):
    if not results:
        return [], [], gr.update(choices=[], value=[]), format_vram_status()
    filtered = results
    if show_type == "Selected":
        filtered = [r for r in filtered if _field(r, "auto_selected", False)]
    elif show_type == "Rejected":
        filtered = [r for r in filtered if not _field(r, "auto_selected", False)]
    filtered = [r for r in filtered if _field(r, "overall_score", 0.0) >= min_score]
    if sort_by == "Highest Score":
        filtered.sort(key=lambda x: _field(x, "overall_score", 0.0), reverse=True)
    elif sort_by == "Best Smile":
        filtered.sort(key=lambda x: _field(x, "smile_score", 0.0), reverse=True)
    gallery_data = [
        (str(_field(r, "image_path", "")).replace("\\", "/"),
         f"{_field(r, 'filename', '')} — {_field(r, 'overall_score', 0.0):.0f} pts")
        for r in filtered
    ]
    df_rows = [
        [_field(r, "filename", ""), round(_field(r, "overall_score", 0.0), 1),
         round(_field(r, "smile_score", 0.0), 1), round(_field(r, "eye_open_score", 0.0), 1),
         round(_field(r, "sharpness_score", 0.0), 1), round(_field(r, "exposure_score", 0.0), 1),
         round(_field(r, "aesthetic_score", 0.0), 1), _field(r, "score_label", ""),
         _field(r, "brief_note", "")]
        for r in filtered
    ]
    choices = [_field(r, "filename", "") for r in filtered]
    current_selected = [_field(r, "filename", "") for r in filtered if _field(r, "auto_selected", False)]
    return gallery_data, df_rows, gr.update(choices=choices, value=current_selected), format_vram_status()


def on_generate_video(selected_paths, slide_duration, transition, mood, bgm_name, volume, progress=gr.Progress()):
    if not selected_paths:
        return None, "❌ No photos selected for video!", None
    transition_map = {"Fade": "fade", "Zoom (Ken Burns)": "zoom", "None": "none"}
    mood_map = {
        "Classic (Warm, nostalgic)": "classic",
        "Pop (Bright, vibrant)": "pop",
        "Cinematic (Dark, moody)": "cinematic",
    }
    bgm_map = {v["name"]: k for k, v in config.BGM_TRACKS.items()}
    video_config = VideoConfig(
        slide_duration=slide_duration,
        transition=transition_map.get(transition, "fade"),
        mood=mood_map.get(mood, "classic"),
        bgm_key=bgm_map.get(bgm_name, "none"),
        bgm_volume=volume,
        output_filename=get_output_filename("studio"),
    )
    generator = VideoGenerator()

    def update_progress(pct, msg):
        progress(pct, desc=msg)

    try:
        output_path = generator.generate(selected_paths, video_config, update_progress)
        if not output_path:
            return None, "❌ Could not generate video from current data.", None
        return output_path, f"✅ Video generated successfully! Saved at: {output_path}", output_path
    except Exception as e:
        return None, f"❌ Error during video generation: {str(e)}", None


custom_css = """
:root {
    --c-bg:          #F7F6F2;
    --c-bg-2:        #F0EEE8;
    --c-surface:     #FFFFFF;
    --c-border:      #E2DED4;
    --c-border-2:    #D0CBC0;
    --c-teal:        #0D9488;
    --c-teal-dark:   #0A7A70;
    --c-teal-pale:   #EEF9F8;
    --c-amber:       #B45309;
    --c-amber-pale:  #FEF3C7;
    --c-text:        #1C2B3A;
    --c-text-2:      #4A5568;
    --c-text-3:      #8896A5;
    --c-success:     #059669;
    --c-success-pale:#ECFDF5;
    --c-danger:      #DC2626;
    --c-danger-pale: #FEF2F2;
    --r:             12px;
    --r-sm:          8px;
    --sh-sm:         0 2px 8px rgba(28,43,58,.07);
    --sh-md:         0 6px 20px rgba(28,43,58,.10);
}

* { 
    box-sizing: border-box; 
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', -apple-system, system-ui, sans-serif !important;
    background: var(--c-bg) !important;
    color: var(--c-text) !important;
}
.gradio-container {
    max-width: 1300px !important;
    margin: 0 auto !important;
    padding: 40px 32px 100px !important;
}

/* ─── Hero ─── */
.studio-hero {
    padding: 52px 0 40px;
    text-align: center;
}
.studio-hero::after {
    content: '';
    display: block;
    width: 64px;
    height: 3px;
    background: linear-gradient(90deg, var(--c-teal), var(--c-amber));
    border-radius: 2px;
    margin: 24px auto 0;
}
.studio-hero h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: clamp(2.3rem, 5.2vw, 3.5rem) !important;
    font-weight: 400 !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 10px;
}
.studio-hero h1 em {
    font-style: italic;
    color: var(--c-teal) !important;
    -webkit-text-fill-color: var(--c-teal) !important;
}
.studio-hero p.subtitle {
    font-size: 1.05rem;
    color: var(--c-text-2) !important;
    max-width: 580px;
    margin: 0 auto 16px;
    line-height: 1.65;
}
.studio-hero .meta {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: 9999px;
    font-size: .78rem;
    color: var(--c-text-3);
    font-weight: 500;
}
.studio-hero .meta .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--c-success);
    box-shadow: 0 0 0 3px var(--c-success-pale);
}

/* ─── Status Row ─── */
.status-row {
    display: flex;
    gap: 16px;
    margin-bottom: 28px !important;
    flex-wrap: wrap;
}
.vram-badge, .live-status,
.vram-badge *, .live-status * {
    background: var(--c-surface) !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
    opacity: 1 !important;
}

.vram-badge, .live-status {
    border: 1px solid var(--c-border) !important;
    border-radius: 9999px !important;
    padding: 12px 20px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    box-shadow: var(--sh-sm) !important;
    text-align: center;
}
.vram-badge p, .live-status p { margin: 0 !important; }

/* ─── Tabs ─── */
.tabs { background: transparent !important; border: none !important; }
.tab-nav, div.tab-nav {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r) !important;
    padding: 8px !important;
    box-shadow: var(--sh-sm) !important;
    margin-bottom: 28px !important;
    display: flex !important;
}
.tab-nav button, div.tab-nav button {
    flex: 1 !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    color: var(--c-text-2) !important;
    background: transparent !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: all .2s ease !important;
}
div.tab-nav button:hover { background: var(--c-bg) !important; color: var(--c-text) !important; }
div.tab-nav button.selected {
    background: var(--c-teal) !important;
    color: #fff !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(13,148,136,.35) !important;
}
.tabitem { background: transparent !important; border: none !important; padding: 4px 0 !important; }

/* ─── Panels ─── */
.gr-panel, .block, .form, .box {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r) !important;
    box-shadow: var(--sh-sm) !important;
}

/* ─── Labels ─── */
label > span:first-child, .block > label > span:first-child,
.wrap > label > span:first-child, fieldset > legend {
    font-size: 0.77rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
    color: var(--c-text-3) !important;
    margin-bottom: 8px !important;
}

/* ─── Inputs ─── */
input[type="text"], input[type="number"], input[type="email"], textarea, select {
    background: var(--c-bg) !important;
    border: 1.5px solid var(--c-border) !important;
    border-radius: var(--r-sm) !important;
    color: var(--c-text) !important;
    font-size: .92rem !important;
    padding: 10px 14px !important;
    transition: border-color .18s, box-shadow .18s !important;
}
input:focus, textarea:focus, select:focus {
    border-color: var(--c-teal) !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,.12) !important;
    background: var(--c-surface) !important;
    outline: none !important;
}
input[type="range"] { accent-color: var(--c-teal) !important; }

/* ─── Radio & Checkbox ─── */
.gr-radio .wrap, .gr-checkbox-group .wrap, .checkbox-group .wrap, 
[data-testid="radio-group"] .wrap, [data-testid="checkbox-group"] .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    width: 100% !important;
}

fieldset .wrap label,
[data-testid="radio-group"] label,
[data-testid="checkbox-group"] label,
.checkbox-group label {
    background: var(--c-surface) !important;
    border: 1.5px solid var(--c-border) !important;
    border-radius: var(--r-sm) !important;
    padding: 16px 24px !important;
    margin: 0 !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    gap: 14px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    user-select: none !important;
    min-height: 60px !important;
    width: 100% !important;
    overflow: visible !important;
}

/* Ensure text inside labels is clear and doesn't wrap awkwardly */
fieldset .wrap label span, 
[data-testid="radio-group"] label span,
[data-testid="checkbox-group"] label span {
    color: var(--c-text) !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    line-height: 1.3 !important;
    text-align: left !important;
    flex-grow: 1 !important;
    white-space: normal !important;
    word-break: break-word !important;
}

fieldset .wrap label input[type="radio"],
fieldset .wrap label input[type="checkbox"],
[data-testid="radio-group"] label input,
[data-testid="checkbox-group"] label input {
    pointer-events: auto !important;
    margin: 0 !important;
    width: 20px !important; 
    height: 20px !important;
    cursor: pointer !important;
    flex-shrink: 0 !important;
}

input[type="radio"]    { accent-color: var(--c-teal) !important; }
input[type="checkbox"] { accent-color: var(--c-amber) !important; }

/* Hover States */
fieldset .wrap label:hover, [data-testid="radio-group"] label:hover {
    border-color: var(--c-teal) !important;
    background: var(--c-teal-pale) !important;
    transform: translateX(4px) !important;
}
[data-testid="checkbox-group"] label:hover, .checkbox-group label:hover {
    border-color: var(--c-amber) !important;
    background: var(--c-amber-pale) !important;
    transform: translateX(4px) !important;
}

/* Selected States */
fieldset .wrap label:has(input:checked),
[data-testid="radio-group"] label:has(input:checked) {
    border-color: var(--c-teal) !important;
    background: var(--c-teal-pale) !important;
    box-shadow: 0 4px 12px rgba(13,148,136,0.12) !important;
}
[data-testid="checkbox-group"] label:has(input:checked),
.checkbox-group label:has(input:checked) {
    border-color: var(--c-amber) !important;
    background: var(--c-amber-pale) !important;
    box-shadow: 0 4px 12px rgba(180,83,9,0.12) !important;
}

/* Bold text when selected */
fieldset .wrap label:has(input:checked) span,
[data-testid="radio-group"] label:has(input:checked) span,
[data-testid="checkbox-group"] label:has(input:checked) span {
    font-weight: 700 !important;
}


/* ─── Buttons ─── */
button.primary, .gr-button-primary {
    font-weight: 700 !important;
    font-size: .95rem !important;
    color: #FFFFFF !important;
    background: var(--c-teal) !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    padding: 14px 30px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 16px rgba(13,148,136,.28) !important;
    transition: all .22s ease !important;
}
button.primary:hover {
    background: var(--c-teal-dark) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 24px rgba(13,148,136,.38) !important;
}
button.primary:active { transform: translateY(0) !important; }
button.secondary, .gr-button-secondary {
    font-weight: 600 !important;
    color: var(--c-teal-dark) !important;
    background: var(--c-surface) !important;
    border: 1.5px solid var(--c-teal) !important;
    border-radius: var(--r-sm) !important;
    padding: 11px 22px !important;
}
button.secondary:hover { background: var(--c-teal-pale) !important; }
button.stop, .gr-button-stop {
    font-weight: 600 !important;
    color: var(--c-danger) !important;
    background: var(--c-danger-pale) !important;
    border: 1.5px solid #FECACA !important;
    border-radius: var(--r-sm) !important;
    padding: 11px 22px !important;
}
button.stop:hover { background: #FEE2E2 !important; border-color: var(--c-danger) !important; }

/* ─── File Upload ─── */
[data-testid="file-upload"], .file-preview, .upload-button, .file-drop {
    background: var(--c-surface) !important;
    border: 2px dashed var(--c-border-2) !important;
    border-radius: var(--r) !important;
    color: var(--c-text-2) !important;
    min-height: 160px !important;
    transition: all .2s ease !important;
}
[data-testid="file-upload"]:hover {
    border-color: var(--c-teal) !important;
    background: var(--c-teal-pale) !important;
}

/* ─── Gallery ─── */
.grid-wrap, .gr-gallery {
    background: var(--c-bg-2) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r) !important;
    padding: 12px !important;
}
.thumbnail-item img, .gr-gallery img {
    border-radius: var(--r-sm) !important;
    box-shadow: var(--sh-sm) !important;
    transition: transform .25s ease, box-shadow .25s ease !important;
}
.thumbnail-item img:hover {
    transform: scale(1.04) !important;
    box-shadow: var(--sh-md) !important;
}

/* ─── Dataframe / Table ─── */
.gr-dataframe, .gr-dataframe > div, .gr-dataframe .table-wrap,
.gradio-dataframe {
    background: var(--c-surface) !important;
    color: var(--c-text) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r) !important;
    overflow: hidden !important;
}
.gr-dataframe table, .gradio-dataframe table {
    background: var(--c-surface) !important;
    color: var(--c-text) !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
    table-layout: auto !important;
}
.gr-dataframe th, .gradio-dataframe th, table th {
    background: var(--c-bg-2) !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
    font-weight: 700 !important;
    font-size: .75rem !important;
    letter-spacing: .05em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid var(--c-border) !important;
    padding: 13px 14px !important;
    text-align: left !important;
    white-space: nowrap !important;
    min-width: 80px !important;
}
.gr-dataframe td, .gradio-dataframe td, table td {
    background: var(--c-surface) !important;
    color: var(--c-text-2) !important;
    -webkit-text-fill-color: var(--c-text-2) !important;
    font-size: .88rem !important;
    border-bottom: 1px solid var(--c-bg-2) !important;
    padding: 11px 14px !important;
    white-space: nowrap !important;
}
.gr-dataframe td input, .gradio-dataframe td input {
    background: transparent !important;
    color: var(--c-text-2) !important;
    border: none !important;
    box-shadow: none !important;
}
.gr-dataframe tr:hover td, .gradio-dataframe tr:hover td {
    background: var(--c-teal-pale) !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
}
.gr-dataframe .table-wrap { overflow-x: auto !important; }

/* ─── Number Metrics (Nuclear Visibility Fix) ─── */
.gr-number, [data-testid="number"], .gr-number *, .gr-slider *, .gr-box * {
    color: #1C2B3A !important;
    -webkit-text-fill-color: #1C2B3A !important;
    opacity: 1 !important;
    visibility: visible !important;
    filter: none !important;
}

.gr-number, [data-testid="number"], .gr-box {
    background: #FFFFFF !important;
    border: 1.5px solid var(--c-border) !important;
    box-shadow: var(--sh-sm) !important;
    overflow: visible !important;
}

/* Specific styling for the numerical value */
.gr-number input,
.gr-number input[type="number"],
.gr-number input[disabled],
.gr-number input[readonly],
[data-testid="number"] input,
.gr-slider input,
.gr-number .input-container input,
.gr-number div.content,
.gr-number p {
    font-size: 2.6rem !important;
    font-weight: 900 !important;
    background: #FFFFFF !important;
    color: #1C2B3A !important;
    border: none !important;
    box-shadow: none !important;
    text-align: center !important;
    min-height: 70px !important;
    line-height: 70px !important;
}


/* Ensure placeholder or empty state is also dark */
.gr-number input::placeholder {
    color: #1C2B3A !important;
}

/* High contrast for the labels */
.gr-number label span, 
[data-testid="number"] label span,
.gr-number .block label span {
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-size: 0.85rem !important;
    margin-bottom: 4px !important;
    display: block !important;
}




/* ─── Accordion ─── */
.gr-accordion > .label-wrap, details > summary {
    background: var(--c-bg) !important;
    color: var(--c-text) !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    border-bottom: 1px solid var(--c-border) !important;
    padding: 14px 18px !important;
    cursor: pointer !important;
}
.gr-accordion > .label-wrap:hover { background: var(--c-teal-pale) !important; }

/* ─── Markdown ─── */
.gr-markdown { color: var(--c-text-2) !important; }
.gr-markdown p { color: var(--c-text-2) !important; line-height: 1.7 !important; }
.gr-markdown strong { color: var(--c-text) !important; font-weight: 700 !important; }
.gr-markdown code {
    font-size: .82rem !important;
    background: var(--c-bg-2) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: 4px !important;
    padding: 2px 7px !important;
    color: var(--c-teal-dark) !important;
}
.gr-markdown h2, .gr-markdown h3 {
    font-family: 'DM Serif Display', serif !important;
    font-weight: 400 !important;
    color: var(--c-text) !important;
}

/* ─── Progress ─── */
.progress-bar { background: linear-gradient(90deg, var(--c-teal), var(--c-teal-dark)) !important; border-radius: 4px !important; }
.progress-track { background: var(--c-border) !important; border-radius: 4px !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: var(--c-bg-2); }
::-webkit-scrollbar-thumb { background: var(--c-border-2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--c-text-3); }

::selection { background: #B2EBE5; color: var(--c-teal-dark); }

.footer-note {
    text-align: center;
    color: var(--c-text-3) !important;
    font-size: .85rem !important;
    margin-top: 24px !important;
    padding: 16px !important;
}

@media (max-width: 768px) {
    .gradio-container { padding: 0 16px 60px !important; }
    .studio-hero { padding: 32px 0 24px; }
    div.tab-nav button { padding: 9px 12px !important; font-size: .82rem !important; }
    .status-row { flex-direction: column; }
    .gr-dataframe th, .gr-dataframe td { font-size: .78rem !important; padding: 8px 10px !important; }
}
"""



# ─────────────────────────────────────────────────────────────────────────────
#  UI Construction
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="VibeFlow Photo Studio",
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "sans-serif"],
    ),
    css=custom_css,
) as demo:

    # Font preload
    gr.HTML("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    """)

    # Hero header
    gr.HTML("""
<div class="studio-hero">
    <h1>📸 VibeFlow <em>Photo Studio</em></h1>
    <p class="subtitle">AI-powered photo analysis & automatic video generation</p>
    <div class="meta"><span class="dot"></span> Optimized for 4 GB VRAM · AI-powered selection</div>
</div>
    """)

    # Status row
    with gr.Row(equal_height=True, elem_classes="status-row"):
        vram_status = gr.Markdown(format_vram_status(), elem_classes="vram-badge")
        live_status = gr.Markdown("⬤ Ready to analyze.", elem_classes="live-status")

    # States
    uploaded_paths_state   = gr.State([])
    analysis_results_state = gr.State([])
    selected_paths_state   = gr.State([])

    with gr.Tabs() as tabs:

        # ─── TAB 1: UPLOAD & CONFIG ───────────────────────────────────────
        with gr.Tab("① Upload & Config", id="upload"):
            file_upload = gr.Files(
                label="Drag & drop images here — JPG · PNG · WEBP",
                file_types=[".jpg", ".jpeg", ".png", ".webp"],
                file_count="multiple",
                height=180,
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    theme_radio = gr.Radio(
                        label="Theme",
                        choices=["Family", "Individual", "Group"],
                        value="Family",
                        interactive=True,
                    )
                with gr.Column(scale=1):
                    priority_check = gr.CheckboxGroup(
                        label="Filter Priority",
                        choices=["Smile", "Quality", "Lighting", "Aesthetic"],
                        value=["Smile", "Quality"],
                        interactive=True,
                    )
                with gr.Column(scale=1):
                    sensitivity_slider = gr.Slider(
                        minimum=5, maximum=10, value=7, step=1,
                        label="AI Sensitivity (higher = stricter)",
                        interactive=True,
                    )
                    photo_count_display = gr.Number(
                        label="Photos Uploaded", interactive=False, value=0,
                    )

            with gr.Accordion("📖 Guide & Score Explanation", open=False):
                gr.Markdown("""
**Sensitivity** — Minimum score = Sensitivity × 10. Default 7 → photos below 70/100 are flagged.

| Metric | Description |
|---|---|
| **Smile** | Mouth width & corner lift ratio (0–100) |
| **Eyes Open** | EAR ratio — warns if eyes are closed |
| **Aesthetic** | CLIP model art quality score (backbone 50%) |
| **Sharpness** | Laplacian edge contrast measure (25%) |
| **Lighting** | Histogram distribution analysis (25%) |
| **Eyes Open** | Stage 1 gate — rejected if < 30 |

`Score = 50% Aesthetic + 25% Sharpness + 25% Lighting` (Bonus +1.5 if smiling)
                """)

            upload_preview = gr.Gallery(label="Photo Preview", columns=6, height=280, object_fit="cover")
            analyze_btn = gr.Button("🤖  Start AI Analysis →", variant="primary", size="lg")

        # ─── TAB 2: ANALYSIS RESULTS ──────────────────────────────────────
        with gr.Tab("② Analysis Results", id="analysis"):
            status_text = gr.Textbox(
                label="Status", interactive=False, lines=1,
                placeholder="Results will appear after analysis...",
            )

            with gr.Row():
                num_selected = gr.Number(label="✅  Selected", interactive=False)
                num_rejected = gr.Number(label="❌  Rejected",    interactive=False)
                avg_score    = gr.Number(label="📈  Avg Score",    interactive=False)
                avg_smile    = gr.Number(label="😊  Avg Smile", interactive=False)

            with gr.Row():
                min_score_filter = gr.Slider(0, 100, value=60, step=5, label="Minimum Score", interactive=True)
                sort_dropdown = gr.Dropdown(
                    label="Sort By",
                    choices=["Highest Score", "Best Smile", "Original Order"],
                    value="Highest Score", interactive=True,
                )
                show_radio = gr.Radio(
                    label="Show",
                    choices=["All", "Selected", "Rejected"],
                    value="All", interactive=True,
                )

            results_gallery = gr.Gallery(
                label="Analysis Results", columns=5, height=380,
                object_fit="cover", allow_preview=True,
            )

            results_table = gr.Dataframe(
                headers=["Photo", "Score", "Smile", "Eyes", "Sharpness",
                         "Lighting", "Aesthetic", "Status", "AI Note"],
                datatype=["str", "number", "number", "number", "number",
                          "number", "number", "str", "str"],
                interactive=False, wrap=True,
            )

            selection_checkbox = gr.CheckboxGroup(
                label="Select photos for video (AI pre-selected)", choices=[],
            )

            proceed_btn = gr.Button("🎬  Set Up Video →", variant="primary")

        # ─── TAB 3: VIDEO BUILDER ─────────────────────────────────────────
        with gr.Tab("③ Build & Export Video", id="video"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    slide_duration = gr.Slider(
                        2.0, 8.0, value=3.0, step=0.5,
                        label="⏱  Duration per photo (seconds)",
                    )
                    transition_radio = gr.Radio(
                        label="✨  Transition Effect",
                        choices=["Fade", "Zoom (Ken Burns)", "None"],
                        value="Fade",
                    )
                with gr.Column(scale=1):
                    mood_radio = gr.Radio(
                        label="🎨  Style",
                        choices=[
                            "Classic (Warm, nostalgic)",
                            "Pop (Bright, vibrant)",
                            "Cinematic (Dark, moody)",
                        ],
                        value="Classic (Warm, nostalgic)",
                    )
                with gr.Column(scale=1):
                    bgm_radio = gr.Radio(
                        label="🎵  Background Music",
                        choices=[config.BGM_TRACKS[k]["name"] for k in config.BGM_TRACKS],
                        value="No Music",
                    )
                    bgm_volume = gr.Slider(0, 1, value=0.8, step=0.1, label="🔊  Volume")

            video_info = gr.Markdown("Please select photos in Tab ② before generating video.")
            generate_btn = gr.Button("🎬  Generate Video", variant="primary", size="lg")
            gen_progress_text = gr.Textbox(label="Progress", interactive=False)
            video_output = gr.Video(label="🎥  Your Video", height=420)

            with gr.Row():
                download_btn = gr.DownloadButton(label="⬇  Download MP4", variant="secondary")
                restart_btn = gr.Button("↩  Start Over", variant="stop")

            gr.HTML('<p class="footer-note">📂 Videos are auto-saved to the <code>output/</code> folder.</p>')

    # ─────────────────────────── Tab navigation JS ───────────────────────
    JS_GOTO_TAB_1 = """() => { setTimeout(() => { const t = document.querySelectorAll('button[role="tab"]'); if(t[0]) t[0].click(); }, 100); }"""
    JS_GOTO_TAB_2 = """() => { setTimeout(() => { const t = document.querySelectorAll('button[role="tab"]'); if(t[1]) t[1].click(); }, 100); }"""
    JS_GOTO_TAB_3 = """() => { setTimeout(() => { const t = document.querySelectorAll('button[role="tab"]'); if(t[2]) t[2].click(); }, 100); }"""

    # ─────────────────────────── Event wiring ───────────────────────
    file_upload.change(
        fn=on_upload,
        inputs=[file_upload],
        outputs=[uploaded_paths_state, photo_count_display, upload_preview],
    )

    analyze_btn.click(
        fn=on_analyze,
        inputs=[uploaded_paths_state, theme_radio, priority_check, sensitivity_slider],
        outputs=[
            analysis_results_state, selected_paths_state, results_gallery, results_table,
            selection_checkbox, num_selected, num_rejected, avg_score, avg_smile,
            live_status, vram_status, status_text,
        ],
        show_progress="full",
    ).then(fn=None, js=JS_GOTO_TAB_2)

    selection_checkbox.change(
        fn=on_selection_change_combined,
        inputs=[selection_checkbox, analysis_results_state, slide_duration],
        outputs=[selected_paths_state, video_info],
    )

    slide_duration.change(
        fn=lambda paths, dur: f"**Video config:** {len(paths)} photos · Estimated ~{len(paths)*dur:.1f} seconds",
        inputs=[selected_paths_state, slide_duration],
        outputs=[video_info],
    )

    proceed_btn.click(fn=None, js=JS_GOTO_TAB_3)

    filter_inputs  = [analysis_results_state, min_score_filter, sort_dropdown, show_radio]
    filter_outputs = [results_gallery, results_table, selection_checkbox, vram_status]
    min_score_filter.change(fn=update_analysis_display, inputs=filter_inputs, outputs=filter_outputs)
    sort_dropdown.change(fn=update_analysis_display, inputs=filter_inputs, outputs=filter_outputs)
    show_radio.change(fn=update_analysis_display, inputs=filter_inputs, outputs=filter_outputs)

    generate_btn.click(
        fn=on_generate_video,
        inputs=[selected_paths_state, slide_duration, transition_radio, mood_radio, bgm_radio, bgm_volume],
        outputs=[video_output, gen_progress_text, download_btn],
    )

    def restart_app():
        clean_temp_dir()
        return [], [], [], None, 0, []

    restart_btn.click(
        fn=restart_app,
        outputs=[
            uploaded_paths_state, analysis_results_state, selected_paths_state,
            file_upload, photo_count_display, upload_preview,
        ],
    ).then(fn=None, js=JS_GOTO_TAB_1)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, status_update_rate=1)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
