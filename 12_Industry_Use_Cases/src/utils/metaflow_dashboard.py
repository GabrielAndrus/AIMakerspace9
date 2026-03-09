import os
import gradio as gr
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

METAFLOW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".metaflow")


def get_run_status(run_dir: Path) -> str:
    end_marker = run_dir / "end"
    if end_marker.exists():
        for f in run_dir.iterdir():
            if f.is_dir():
                task_dir = f
                for tf in task_dir.iterdir():
                    if tf.name == "0.task_ok":
                        return "✓"
                    if tf.name == "0.task_end":
                        return "✗"
        return "✓"
    
    start_marker = run_dir / "start"
    if start_marker.exists():
        return "✅"
    return "⏳"


def get_flow_names() -> list[str]:
    try:
        metaflow_path = Path(METAFLOW_DIR)
        if not metaflow_path.exists():
            return ["No flows found"]
        
        flows = []
        for item in metaflow_path.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                flows.append(item.name)
        
        if not flows:
            return ["No flows found"]
        return sorted(flows)
    except Exception:
        return ["No flows found"]


def get_runs_for_flow(flow_name: str) -> list[dict]:
    if not flow_name or flow_name == "No flows found":
        return []
    
    try:
        flow_path = Path(METAFLOW_DIR) / flow_name
        if not flow_path.exists():
            return []
        
        runs = []
        for run_dir in flow_path.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            
            try:
                status = get_run_status(run_dir)
                
                meta_path = run_dir / "_meta" / "_self.json"
                started = "-"
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                            created_at = meta.get("ts", 0)
                            if created_at:
                                started = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass
                
                if started == "-":
                    try:
                        created_at = int(run_dir.name[:10])
                        started = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass
                
                duration = "-"
                
                runs.append({
                    "run_id": run_dir.name,
                    "status": status,
                    "started": started,
                    "duration": duration,
                    "user": "unknown"
                })
            except Exception:
                continue
        
        runs = sorted(runs, key=lambda r: r["started"], reverse=True)[:20]
        return runs
    except Exception:
        return []


def get_steps_for_run(flow_name: str, run_id: str) -> list[dict]:
    if not flow_name or not run_id:
        return []
    
    try:
        run_path = Path(METAFLOW_DIR) / flow_name / run_id
        if not run_path.exists():
            return []
        
        steps = []
        step_names = ["start", "load_data", "validate_data", "preprocess", "train_model", "evaluate", "save_model", "end"]
        
        available_steps = [d.name for d in run_path.iterdir() if d.is_dir() and not d.name.startswith("_")]
        
        for step_name in available_steps:
            step_dir = run_path / step_name
            status = "⏳"
            task_count = 0
            
            for task_dir in step_dir.iterdir():
                if task_dir.is_dir() and task_dir.name.isdigit():
                    task_count += 1
                    task_ok = task_dir / "0.task_ok"
                    task_end = task_dir / "0.task_end"
                    
                    if task_ok.exists():
                        status = "✓"
                    elif task_end.exists():
                        status = "✗"
            
            error_msg = ""
            if status == "✗":
                for task_dir in step_dir.iterdir():
                    if task_dir.is_dir():
                        stderr = task_dir / "0.runtime_stderr.log"
                        if stderr.exists():
                            try:
                                with open(stderr) as f:
                                    content = f.read()
                                    if content:
                                        error_msg = content[:200]
                                        break
                            except Exception:
                                pass
            
            steps.append({
                "step_name": step_name,
                "status": status,
                "task_count": task_count,
                "error": error_msg
            })
        
        return steps
    except Exception:
        return []


def build_runs_table(flow_name: str) -> list[list]:
    runs = get_runs_for_flow(flow_name)
    if not runs:
        return [["No runs found", "-", "-", "-", "-"]]
    
    return [
        [r["run_id"], r["status"], r["started"], r["duration"], r["user"]]
        for r in runs
    ]


def build_steps_table(flow_name: str, run_id: str) -> list[list]:
    steps = get_steps_for_run(flow_name, run_id)
    if not steps:
        return [["No steps found", "-", "-", "-"]]
    
    return [
        [s["step_name"], s["status"], s["task_count"], s["error"] if s["error"] else "-"]
        for s in steps
    ]


def refresh_dashboard(flow_name: str) -> tuple:
    table_data = build_runs_table(flow_name)
    flow_names = get_flow_names()
    return gr.update(choices=flow_names), table_data


def on_flow_select(flow_name: str) -> tuple:
    table_data = build_runs_table(flow_name)
    return table_data, gr.update(value=None)


def on_run_select(flow_name: str, run_id: str) -> tuple:
    steps_data = build_steps_table(flow_name, run_id)
    return steps_data


def create_dashboard():
    flow_names = get_flow_names()
    
    with gr.Blocks(title="Metaflow Dashboard", theme="JohnSmith9982/small_and_pretty") as dashboard:
        gr.Markdown("# Metaflow Dashboard")
        gr.Markdown("Real-time monitoring of your Metaflow training runs (auto-refreshes every 3s)")
        
        with gr.Row():
            flow_dropdown = gr.Dropdown(
                choices=flow_names,
                label="Select Flow",
                value=flow_names[0] if flow_names else None,
                interactive=True
            )
            refresh_btn = gr.Button("🔄 Refresh")
        
        gr.Markdown("## Runs")
        runs_table = gr.Dataframe(
            headers=["Run ID", "Status", "Started", "Duration", "User"],
            value=build_runs_table(flow_names[0]) if flow_names else [["No flows found", "-", "-", "-", "-"]],
            interactive=False,
            wrap=True,
            max_height=300
        )
        
        with gr.Row():
            run_id_input = gr.Textbox(label="Run ID", placeholder="Select a run from table or enter ID")
            show_steps_btn = gr.Button("Show Steps")
        
        gr.Markdown("## Steps")
        steps_table = gr.Dataframe(
            headers=["Step", "Status", "Tasks", "Error"],
            value=[],
            interactive=False,
            wrap=True,
            max_height=400
        )
        
        flow_dropdown.change(
            on_flow_select,
            inputs=[flow_dropdown],
            outputs=[runs_table, run_id_input]
        )
        
        refresh_btn.click(
            refresh_dashboard,
            inputs=[flow_dropdown],
            outputs=[flow_dropdown, runs_table]
        )
        
        show_steps_btn.click(
            on_run_select,
            inputs=[flow_dropdown, run_id_input],
            outputs=[steps_table]
        )
        
        runs_table.select(
            lambda evt, fn=flow_dropdown: on_run_select(fn, evt.selected_row[0])[0],
            inputs=[flow_dropdown],
            outputs=[steps_table]
        )
        
        timer = gr.Timer(3)
        timer.tick(
            refresh_dashboard,
            inputs=[flow_dropdown],
            outputs=[flow_dropdown, runs_table]
        )
    
    return dashboard


def main():
    dashboard = create_dashboard()
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=3001,
        share=False
    )


if __name__ == "__main__":
    main()
