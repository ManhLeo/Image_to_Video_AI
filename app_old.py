"""Gradio entrypoint for AI Photo Studio end-to-end workflow."""

import gradio as gr
from services.batch_processor import BatchProcessor
from services.video_generator import VideoGenerator, VideoConfig
from utils.file_utils import get_output_filename, clean_temp_dir
from utils.system_utils import format_vram_status
import config


# Global state structure
initial_state = {
    "uploaded_paths": [],
    "user_config": {},
    "analysis_results": [],
    "selected_paths": [],
    "video_path": None,
}


def _field(item, key, default=None):
    """Read value from either dict payload or dataclass-like object."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _result_to_payload(result):
    """Convert analysis result object to JSON-serializable payload."""
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
    }

def on_upload(files):
    """Handles file uploads."""
    if not files:
        return [], 0, []
    paths = [f.name for f in files]
    return paths, len(paths), paths

def on_analyze(uploaded_paths, theme, priorities, sensitivity, progress=gr.Progress()):
    """Runs AI analysis on the uploaded photos."""
    if not uploaded_paths:
        raise gr.Error("Bạn chưa tải ảnh nào lên!")
    
    # Mapping Vietnamese UI labels to internal keys
    theme_map = {
        "gia đình": "family",
        "cá nhân": "individual",
        "nhóm": "group"
    }
    priority_map = {
        "nụ cười": "smile",
        "chất lượng": "quality",
        "ánh sáng": "lighting",
        "thẩm mỹ": "aesthetic"
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
            desc = f"Đang phân tích ảnh {current}/{total}: {latest_result.filename} — {latest_result.score_label}"
        else:
            desc = f"Đang xử lý... {current}/{total}"
        progress(progress_pct, desc=desc)

    
    results = processor.process_all(
        uploaded_paths,
        user_config,
        progress_callback=update_progress
    )
    
    summary = processor.get_summary()
    processor.close()
    
    payload_results = [_result_to_payload(r) for r in results]

    # Gallery data: Tuple (path, caption) - Gradio 6.13.0 requires this
    gallery_data = [
        (str(_field(r, "image_path", "")).replace("\\", "/"), f"{_field(r, 'filename', '')} - {int(_field(r, 'overall_score', 0))} pts")
        for r in payload_results
    ]
    
    # Dataframe rows - EXPLICIT CAST & Path sanitization
    df_rows = [
        [
            str(r["filename"]), 
            float(round(r["overall_score"], 1)), 
            float(round(r["smile_score"], 1)), 
            float(round(r["eye_open_score"], 1)),
            float(round(r["sharpness_score"], 1)), 
            float(round(r["exposure_score"], 1)), 
            float(round(r["aesthetic_score"], 1)),
            str(r["score_label"]), 
            str(r["brief_note"])
        ]
        for r in payload_results
    ]

    
    try:
        # Deduplicate filenames for CheckboxGroup (Surgical Fix)
        all_filenames = list(dict.fromkeys([str(r["filename"]) for r in payload_results]))
        # AI-guided selection: Only pick photos marked as auto_selected
        selected_filenames = [str(r["filename"]) for r in payload_results if r.get("auto_selected", False)]
        
        # Selected paths: Also filter to match the initial selection
        selected_paths = [str(r["image_path"]).replace("\\", "/") for r in payload_results if r.get("auto_selected", False)]
        
        print(f"DEBUG: Analysis complete. Results: {len(results)}")
        if gallery_data:
            print(f"DEBUG: Gallery item type: {type(gallery_data[0])}")
        
        return (
            payload_results,               # analysis_results_state (Index 0 now)
            selected_paths,                # selected_paths_state (1)
            gallery_data,                  # results_gallery (2)
            df_rows,                       # results_table (3)
            gr.update(choices=all_filenames, value=selected_filenames), # selection_checkbox (4)
            int(summary.get("selected_count", 0)), # num_selected (5)
            int(summary.get("rejected_count", 0)), # num_rejected (6)
            float(round(summary.get("avg_score", 0), 1)), # avg_score (7)
            float(round(summary.get("avg_smile", 0), 1)), # avg_smile (8)
            f"Done! Processed {len(results)} photos.", # live_status (9)
            format_vram_status(), # vram_status (10)
            "✅ Analysis complete." # status_text (11)
        )
    except Exception as e:
        print(f"ERROR in on_analyze result preparation: {e}")
        import traceback
        traceback.print_exc()
        raise e


def on_selection_change_combined(selected_filenames, analysis_results, slide_duration):
    """Updates selected paths and video info in one go."""
    if not selected_filenames:
        return [], "Chưa chọn ảnh nào!"

    selected_paths = []
    for filename in selected_filenames:
        for r in analysis_results:
            if _field(r, "filename", "") == filename:
                selected_paths.append(_field(r, "image_path", ""))
    
    count = len(selected_paths)
    total_time = count * slide_duration
    info_msg = f"**Cấu hình video:** {count} ảnh · Ước tính ~{total_time:.1f} giây" if count > 0 else "Chưa chọn ảnh nào!"
    
    return selected_paths, info_msg

def update_analysis_display(results, min_score, sort_by, show_type):
    """Dynamically filters and sorts the analysis results displayed in the UI."""
    if not results:
        return [], [], gr.update(choices=[], value=[]), format_vram_status()
    
    # 1. Filter
    filtered = results
    if show_type == "Được chọn":
        filtered = [r for r in filtered if _field(r, "auto_selected", False)]
    elif show_type == "Bị loại":
        filtered = [r for r in filtered if not _field(r, "auto_selected", False)]
    
    # Filter by min score (strict UI filter)
    filtered = [r for r in filtered if _field(r, "overall_score", 0.0) >= min_score]
    
    # 2. Sort
    if sort_by == "Điểm cao nhất":
        filtered.sort(key=lambda x: _field(x, "overall_score", 0.0), reverse=True)
    elif sort_by == "Nụ cười tốt nhất":
        filtered.sort(key=lambda x: _field(x, "smile_score", 0.0), reverse=True)
    
    # 3. Prepare Gallery data - TUPLE FORMAT
    gallery_data = [
        (str(_field(r, "image_path", "")).replace("\\", "/"), f"{_field(r, 'filename', '')} - {_field(r, 'overall_score', 0.0):.0f} pts")
        for r in filtered
    ]
    
    # 4. Prepare Table data
    df_rows = [
        [str(_field(r, "filename", "")), float(round(_field(r, "overall_score", 0.0), 1)), float(round(_field(r, "smile_score", 0.0), 1)), float(round(_field(r, "eye_open_score", 0.0), 1)),
         float(round(_field(r, "sharpness_score", 0.0), 1)), float(round(_field(r, "exposure_score", 0.0), 1)), float(round(_field(r, "aesthetic_score", 0.0), 1)),
         str(_field(r, "score_label", "")), str(_field(r, "brief_note", ""))]
        for r in filtered
    ]
    
    # 5. Update Selection Checkbox - AI RECOMMENDED ONLY
    choices = [str(_field(r, "filename", "")) for r in filtered]
    current_selected = [str(_field(r, "filename", "")) for r in filtered if _field(r, "auto_selected", False)]
    
    return (gallery_data, df_rows, gr.update(choices=choices, value=current_selected), format_vram_status())

def on_generate_video(selected_paths, slide_duration, transition, mood, bgm_name, volume, progress=gr.Progress()):
    """Generates the final video slideshow."""
    if not selected_paths:
        return None, "❌ Bạn chưa chọn ảnh nào để dựng video!", None
    
    # Map display names back to keys
    transition_map = {"Fade (Mờ dần)": "fade", "Zoom (Ken Burns)": "zoom", "Không có / None": "none"}
    mood_map = {
        "Classic (Ấm áp, hoài niệm)": "classic", 
        "Pop (Tươi sáng, rực rỡ)": "pop", 
        "Cinematic (Điện ảnh, tối)": "cinematic"
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
            return None, "❌ Không thể tạo video từ dữ liệu hiện tại.", None
        return output_path, f"✅ Video đã tạo thành công! Lưu tại: {output_path}", output_path
    except Exception as e:
        return None, f"❌ Lỗi khi tạo video: {str(e)}", None


# Custom CSS for Premium Glassmorphism UI
custom_css = """
* { font-family: 'Outfit', sans-serif !important; }

body {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
    background-attachment: fixed !important;
    color: #e2e8f0 !important;
}

.gradio-container { background: transparent !important; }

/* Glassmorphism for panels */
.form, .box, .panel {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
}

/* Tabs: Keep them simple to avoid JS errors */
.tabs { background: transparent !important; border: none !important; }
.tabitem { background: rgba(30, 41, 59, 0.4) !important; border-radius: 12px !important; }

/* Typography Gradients */
h1, h2, h3 {
    background: linear-gradient(to right, #c084fc, #ec4899) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
}

/* Premium Buttons */
button.primary {
    background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%) !important;
    border: none !important;
    color: white !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(236, 72, 153, 0.3) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(236, 72, 153, 0.5) !important;
}

/* Radio & Checkbox Glow Effect */
.gr-input-label { 
    cursor: pointer !important; 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.gr-input-label:hover { 
    background: rgba(168, 85, 247, 0.1) !important; 
    border-color: rgba(168, 85, 247, 0.3) !important;
}

/* Specific styling for the selected state in Gradio */
.gr-input-label.selected, [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.25) 0%, rgba(236, 72, 153, 0.25) 100%) !important;
    border-color: #f472b6 !important;
    box-shadow: 0 0 15px rgba(236, 72, 153, 0.4), inset 0 0 10px rgba(168, 85, 247, 0.2) !important;
    transform: scale(1.02) !important;
}

input[type="radio"]:checked + span, input[type="checkbox"]:checked + span {
    color: #f472b6 !important;
    font-weight: 700 !important;
    text-shadow: 0 0 8px rgba(236, 72, 153, 0.4) !important;
}

input[type="radio"], input[type="checkbox"] {
    accent-color: #ec4899 !important;
}

/* Inputs */
input, textarea, .dropdown-menu {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
}

input:focus, textarea:focus {
    border-color: #a855f7 !important;
    box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.2) !important;
}

/* Tables */
table { border-radius: 12px !important; overflow: hidden !important; }
th { background: rgba(30, 41, 59, 0.9) !important; color: #c084fc !important; border: none !important; }
td { background: rgba(15, 23, 42, 0.4) !important; border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important; }
"""

# UI Construction
with gr.Blocks(title="AI Photo Studio", theme=gr.themes.Base(), css=custom_css) as demo:
    # Load external fonts via HTML link to avoid CSS @import restrictions in some browsers
    gr.HTML("<link href='https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap' rel='stylesheet'>")
    
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 0;'>📸 VibeFlow Photo Studio</h1><p style='text-align: center; color: #cbd5e1; font-size: 1.1em; margin-top: 5px;'>Hệ thống phân tích ảnh thông minh & dựng video tự động (Optimized for 4GB VRAM)</p>")
    
    with gr.Row():
        vram_status = gr.Markdown(format_vram_status(), elem_classes="vram-alert")
        live_status = gr.Markdown("Sẵn sàng phân tích ảnh.")

    
    # State management
    uploaded_paths_state = gr.State([])
    analysis_results_state = gr.State([])
    selected_paths_state = gr.State([])
    
    tabs = gr.Tabs()
    with tabs:
        # TAB 1: UPLOAD & CONFIG
        with gr.Tab("1. Tải ảnh & Cấu hình", id="upload"):
            file_upload = gr.Files(
                label="📁 Kéo ảnh vào đây / Drag & drop photos",
                file_types=[".jpg", ".jpeg", ".png", ".webp"],
                file_count="multiple",
                height=200
            )
            
            with gr.Row():
                with gr.Column():
                    theme_radio = gr.Radio(
                        label="🎯 Chủ đề / Theme",
                        choices=["Gia đình / Family", "Cá nhân / Individual", "Nhóm / Group"],
                        value="Gia đình / Family",
                        interactive=True
                    )
                    priority_check = gr.CheckboxGroup(
                        label="⭐ Ưu tiên lọc / Filter priority",
                        choices=["Nụ cười / Smile", "Chất lượng / Quality", "Ánh sáng / Lighting", "Thẩm mỹ / Aesthetic"],
                        value=["Nụ cười / Smile", "Chất lượng / Quality"],
                        interactive=True
                    )

                
                with gr.Column():
                    sensitivity_slider = gr.Slider(
                        minimum=5, maximum=10, value=7, step=1,
                        label="🎛 Độ nhạy AI / Sensitivity (cao = lọc chặt hơn)",
                        interactive=True
                    )
                    photo_count_display = gr.Number(label="📷 Số ảnh đã tải lên", interactive=False, value=0)
                    #gr.Markdown("**Lưu ý:** Sensitivity 7 = điểm tối thiểu 70/100. AI sẽ tự động loại bỏ các ảnh không đạt yêu cầu.")
            
            with gr.Accordion("📖 Hướng dẫn & Giải thích chỉ số", open=False):
                gr.Markdown("""
                ### 🎛 Độ nhạy AI (Sensitivity)
                - **Sensitivity 7 (Mặc định)**: Tương ứng với điểm sàn là **70/100**. Ảnh dưới 70 điểm sẽ bị đánh dấu 'Loại bỏ'.
                - **Cách tính**: Điểm sàn = Sensitivity * 10. Các ảnh bị loại sẽ không được tự động chọn cho video.
                
                ### 📊 Các chỉ số đo lường
                - **Nụ cười (Smile)**: Phân tích độ rộng khuôn miệng và độ cong khóe môi (0-100).
                - **Mắt mở (Eye Open)**: Tỷ lệ mở của mí mắt (EAR). Cảnh báo nếu nhắm mắt.
                - **Độ nét (Sharpness)**: Sử dụng thuật toán Laplacian để đo độ tương phản cạnh.
                - **Ánh sáng (Exposure)**: Đo độ sáng trung bình và phân bố histogram.
                - **Thẩm mỹ (Aesthetic)**: CLIP model đánh giá độ 'đẹp' và nghệ thuật của bức ảnh.
                
                ### 🎯 Trọng số tính điểm (Overall Score)
                Điểm tổng kết được tính theo công thức:
                `Score = 25% Nụ cười + 20% Độ nét + 20% Ánh sáng + 20% Thẩm mỹ + 15% Bố cục`
                """)
            
            upload_preview = gr.Gallery(
                label="Xem trước ảnh / Preview",
                columns=5,
                height=300,
                object_fit="cover"
            )
            
            analyze_btn = gr.Button("🤖 Bắt đầu phân tích AI →", variant="primary", size="lg")
            
        # TAB 2: ANALYSIS RESULTS
        with gr.Tab("2. Kết quả phân tích", id="analysis"):
            status_text = gr.Textbox(
                label="📊 Trạng thái / Status",
                interactive=False,
                lines=1,
                placeholder="Kết quả sẽ hiển thị sau khi phân tích..."
            )
            
            with gr.Row():
                num_selected = gr.Number(label="✅ Ảnh được chọn", interactive=False)
                num_rejected = gr.Number(label="❌ Bị loại", interactive=False)
                avg_score = gr.Number(label="📈 Điểm TB", interactive=False)
                avg_smile = gr.Number(label="😊 Nụ cười TB", interactive=False)
            
            with gr.Row():
                min_score_filter = gr.Slider(
                    minimum=0, maximum=100, value=60, step=5,
                    label="Điểm tối thiểu / Min score",
                    interactive=True
                )
                sort_dropdown = gr.Dropdown(
                    label="Sắp xếp / Sort by",
                    choices=["Điểm cao nhất", "Nụ cười tốt nhất", "Thứ tự gốc"],
                    value="Điểm cao nhất",
                    interactive=True
                )
                show_radio = gr.Radio(
                    label="Hiển thị / Show",
                    choices=["Tất cả", "Được chọn", "Bị loại"],
                    value="Tất cả",
                    interactive=True
                )
            
            results_gallery = gr.Gallery(
                label="Kết quả phân tích / Analysis results",
                columns=5,
                height=400,
                object_fit="cover",
                allow_preview=True
            )
            
            results_table = gr.Dataframe(
                headers=["Ảnh", "Điểm tổng", "Nụ cười", "Mắt mở", "Độ nét", "Ánh sáng", "Thẩm mỹ", "Trạng thái", "Ghi chú AI"],
                datatype=["str","number","number","number","number","number","number","str","str"],
                interactive=False,
                wrap=True
            )

            
            selection_checkbox = gr.CheckboxGroup(
                label="✅ Chọn ảnh cho video (đã chọn tự động theo AI)",
                choices=[]
            )
            
            proceed_btn = gr.Button("🎬 Bước tiếp theo: Thiết lập video →", variant="primary")
            
        # TAB 3: VIDEO BUILDER
        with gr.Tab("3. Dựng video & Xuất", id="video"):
            with gr.Row():
                with gr.Column():
                    slide_duration = gr.Slider(
                        minimum=2.0, maximum=8.0, value=3.0, step=0.5,
                        label="⏱ Thời gian mỗi ảnh (giây)"
                    )
                    transition_radio = gr.Radio(
                        label="✨ Hiệu ứng chuyển / Transition",
                        choices=["Fade (Mờ dần)", "Zoom (Ken Burns)", "Không có / None"],
                        value="Fade (Mờ dần)"
                    )
                
                with gr.Column():
                    mood_radio = gr.Radio(
                        label="🎨 Phong cách / Mood",
                        choices=["Classic (Ấm áp, hoài niệm)", "Pop (Tươi sáng, rực rỡ)", "Cinematic (Điện ảnh, tối)"],
                        value="Classic (Ấm áp, hoài niệm)"
                    )
                
                with gr.Column():
                    bgm_radio = gr.Radio(
                        label="🎵 Nhạc nền / Background music",
                        choices=list(config.BGM_TRACKS[k]["name"] for k in config.BGM_TRACKS),
                        value="Không có nhạc"
                    )
                    bgm_volume = gr.Slider(minimum=0, maximum=1, value=0.8, step=0.1, label="🔊 Âm lượng")
            
            video_info = gr.Markdown("Vui lòng chọn ảnh ở Tab 2 trước khi tạo video.")
            
            generate_btn = gr.Button("🎬 Tạo Video", variant="primary", size="lg")
            
            gen_progress_text = gr.Textbox(label="Tiến trình / Progress", interactive=False)
            
            video_output = gr.Video(label="🎥 Video của bạn / Your video", height=400)
            
            with gr.Row():
                download_btn = gr.DownloadButton(label="⬇ Tải xuống MP4", variant="secondary")
                restart_btn = gr.Button("↩ Làm lại từ đầu", variant="stop")
            
            gr.Markdown("📂 Video sau khi tạo sẽ được lưu tự động tại thư mục `output/`.")

    # EVENT HANDLERS
    
    # Upload events
    file_upload.change(
        fn=on_upload,
        inputs=[file_upload],
        outputs=[uploaded_paths_state, photo_count_display, upload_preview]
    )
    
    # Robust JS tab switching (handles Gradio 6 dynamic DOM)
    JS_GOTO_TAB_1 = """() => { 
        setTimeout(() => {
            const tabs = document.querySelectorAll('button.selected, button[role="tab"]');
            if (tabs && tabs[0]) tabs[0].click();
        }, 100);
    }"""
    JS_GOTO_TAB_2 = """() => { 
        setTimeout(() => {
            const tabs = document.querySelectorAll('button[role="tab"]');
            if (tabs && tabs[1]) tabs[1].click();
        }, 100);
    }"""
    JS_GOTO_TAB_3 = """() => { 
        setTimeout(() => {
            const tabs = document.querySelectorAll('button[role="tab"]');
            if (tabs && tabs[2]) tabs[2].click();
        }, 100);
    }"""

    analyze_btn.click(
        fn=on_analyze,
        inputs=[uploaded_paths_state, theme_radio, priority_check, sensitivity_slider],
        outputs=[
            analysis_results_state, 
            selected_paths_state,   
            results_gallery,        
            results_table,          
            selection_checkbox,     
            num_selected,           
            num_rejected,           
            avg_score,              
            avg_smile,              
            live_status,            
            vram_status,            
            status_text             
        ],
        show_progress="full"
    ).then(fn=None, js=JS_GOTO_TAB_2)

    
    # Selection change events
    selection_checkbox.change(
        fn=on_selection_change_combined,
        inputs=[selection_checkbox, analysis_results_state, slide_duration],
        outputs=[selected_paths_state, video_info]
    )
    
    slide_duration.change(
        fn=lambda paths, dur: f"**Cấu hình video:** {len(paths)} ảnh · Ước tính ~{len(paths)*dur:.1f} giây",
        inputs=[selected_paths_state, slide_duration],
        outputs=[video_info]
    )
    

    # Move to Tab 3 button (Video)
    proceed_btn.click(
        fn=None,
        js=JS_GOTO_TAB_3
    )
    
    # Filtering and Sorting events
    filter_inputs = [analysis_results_state, min_score_filter, sort_dropdown, show_radio]
    filter_outputs = [results_gallery, results_table, selection_checkbox, vram_status]
    
    min_score_filter.change(fn=update_analysis_display, inputs=filter_inputs, outputs=filter_outputs)
    sort_dropdown.change(fn=update_analysis_display, inputs=filter_inputs, outputs=filter_outputs)
    show_radio.change(fn=update_analysis_display, inputs=filter_inputs, outputs=filter_outputs)
    
    # Video generation events
    generate_btn.click(
        fn=on_generate_video,
        inputs=[selected_paths_state, slide_duration, transition_radio, mood_radio, bgm_radio, bgm_volume],
        outputs=[video_output, gen_progress_text, download_btn]
    )

    
    
    # Restart event
    def restart_app():
        clean_temp_dir()
        return (
            [], [], [], # reset states
            None, # reset upload
            0, # reset count
            [] # reset gallery
        )
    
    restart_btn.click(
        fn=restart_app,
        outputs=[uploaded_paths_state, analysis_results_state, selected_paths_state, file_upload, photo_count_display, upload_preview]
    ).then(fn=None, js=JS_GOTO_TAB_1)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, status_update_rate=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
