#!/usr/bin/env python3
"""
BFCL Rollout Viewer - Interactive TUI for viewing slime rollout data.

This tool provides an interactive terminal interface for browsing rollout data
saved by slime's --dump-details option. It supports:
- Loading .pt files (slime format)
- Interactive step/sample navigation
- Field filtering
- Search functionality
- BFCL-specific metrics display

Usage:
    python scripts/rollout_viewer.py --rollout-dir /path/to/rollout_data

Requirements:
    pip install textual==0.52.1 rich aiofiles
"""

import asyncio
import json
import traceback
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from rich.highlighter import ReprHighlighter
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text


def check_textual_version():
    """Check if textual version is compatible."""
    try:
        import textual
        from packaging.version import Version

        if Version(textual.__version__) < Version("0.50.0"):
            print(f"Warning: Textual version {textual.__version__} may not be fully compatible.")
            print("Recommended: pip install textual==0.52.1")
    except ImportError:
        raise ImportError("Please install textual: pip install textual==0.52.1")


check_textual_version()

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, ProgressBar, Select, SelectionList, Static

INDEX_KEY = "__IDX"
FILE_SUFFIX = ".pt"

# Fields to display by default (in order)
DEFAULT_DISPLAY_FIELDS = [
    "group_index",
    "index",
    "prompt",
    "response",
    "response_length",
    "reward",
    "status",
    "label",
    "metadata",
]


def load_pt_file(path: Path) -> list[dict]:
    """Load a .pt file and return list of sample dicts."""
    pack = torch.load(path, map_location="cpu", weights_only=False)
    samples = pack.get("samples", [])

    # Process each sample for display
    processed = []
    for idx, sample in enumerate(samples):
        d = {INDEX_KEY: idx}
        for k, v in sample.items():
            if isinstance(v, str):
                d[k] = v
            elif isinstance(v, (list, dict)):
                d[k] = json.dumps(v, ensure_ascii=False, indent=2)
            elif hasattr(v, "name"):  # Enum like Status
                d[k] = v.name
            elif isinstance(v, torch.Tensor):
                if v.numel() <= 10:
                    d[k] = str(v.tolist())
                else:
                    d[k] = f"Tensor(shape={list(v.shape)}, dtype={v.dtype})"
            else:
                d[k] = str(v)
        processed.append(d)
    return processed


async def load_path_async(path: Path, data: dict, idx: int, pbar):
    """Load a single .pt file asynchronously."""
    loop = asyncio.get_event_loop()
    samples = await loop.run_in_executor(None, load_pt_file, path)
    data[idx] = {"samples": samples, "path": str(path)}
    print(f"Loaded: {path.name} ({len(samples)} samples)")
    pbar.advance(1)


async def load_dir_async(path: Path, data: dict, pbar):
    """Load all .pt files in directory asynchronously."""
    paths = sorted(path.glob(f"*{FILE_SUFFIX}"), key=lambda x: x.stem)
    tasks = [load_path_async(p, data, i, pbar) for i, p in enumerate(paths)]
    await asyncio.gather(*tasks)


class Highlighter(ReprHighlighter):
    """Custom highlighter for special tokens and brackets."""

    highlights = ReprHighlighter.highlights + [
        r"(?P<tag_name>[][<>{}()|])",
        r"<\|(?P<tag_name>[\w\W]*?)\|>",
        r"(?P<tool_call><tool_call>.*?</tool_call>)",
    ]


def center_word_with_equals(word: str, total_length: int, char: str = "=") -> str:
    """Center a word with padding characters."""
    if len(word) > total_length:
        return word
    padding = total_length - len(word)
    left_pad = padding // 2
    right_pad = (padding + 1) // 2
    return char * left_pad + " " + word + " " + char * right_pad


def highlight_keyword(content: str, keyword: Optional[str]) -> Text:
    """Highlight keyword occurrences in content."""
    if not keyword:
        return Text(content)
    text = Text()
    parts = content.split(keyword)
    for i, part in enumerate(parts):
        text.append(part)
        if i < len(parts) - 1:
            text.append(keyword, style="on #8f51b5")
    return text


HELP_DOC = """
Keybinds:

- `f/esc`: find/cancel
- `tab/left/right`: change focus
- `j/k`: page down/up
- `g/G`: scroll home/end
- `n/N`: next sample/step
- `p/P`: previous sample/step
- `s`: switch display mode
  - plain text
  - rich table

"""


class RolloutViewer(App):
    """Interactive TUI for viewing slime rollout data."""

    BINDINGS = [
        ("left", "focus_previous", "Focus Previous"),
        ("right", "focus_next", "Focus Next"),
        ("s", "switch_render", "Switch Render"),
        ("n", "next_sample", "Next Sample"),
        ("N", "next_step", "Next Step"),
        ("p", "previous_sample", "Previous Sample"),
        ("P", "previous_step", "Previous Step"),
        ("f", "toggle_search", "Find"),
        ("enter", "next_search", "Find Next"),
        ("escape", "cancel_search", "Cancel Find"),
        ("j", "page_down", "Page Down"),
        ("k", "page_up", "Page Up"),
        ("g", "page_home", "Page Home"),
        ("G", "page_end", "Page End"),
    ]

    CSS = """
    Select:focus > SelectCurrent {
        border: tall #8f51b5;
    }
    Select.-expanded > SelectCurrent {
        border: tall #8f51b5;
    }
    #select-container {
        width: 18%;
        height: 100%;
        align: center top;
    }
    #search-container {
        height: 10%;
        align: center top;
    }
    #search-box {
        width: 60%;
    }
    #index-box {
        width: 40%;
    }
    """

    def __init__(self, step_num: int, data: dict[int, dict], pbar):
        super().__init__()
        self.step_num = step_num
        self.data = data
        self.render_table = False
        self.selected_step_index = 0
        self.selected_sample_index = 0
        self.pbar = pbar
        self.matches = []
        self.current_match_index = 0
        self.highlighter = Highlighter()

        # Initialize field filter from first sample
        first_samples = data[list(data.keys())[0]]["samples"]
        self.filter_fields = [(f, f, True) for f in first_samples[0].keys()]
        self._field_set: set[str] = set(first_samples[0].keys())
        self.sample_num = len(first_samples)

    def compose(self) -> ComposeResult:
        with Horizontal(id="search-container"):
            yield Input(placeholder="Search content...", id="search-box")
            yield Input(placeholder="Jump to index...", id="index-box")
            with Vertical(id="search-container2"):
                yield self.pbar
                yield Static("", id="search-status")

        with Horizontal():
            with Vertical(id="select-container"):
                yield Static("\n")
                yield Static(renderable=Markdown(HELP_DOC), markup=False)
                yield Static("\n")
                yield Select(
                    id="step-select",
                    value=0,
                    prompt="Select Step",
                    options=[("step: 1", 0)],
                    allow_blank=False,
                )
                yield Select(
                    id="sample-select",
                    value=0,
                    prompt="Select Sample",
                    options=[("sample: 1", 0)],
                    allow_blank=False,
                )
                yield Select(
                    id="sample-sort",
                    value=0,
                    prompt="Sort By",
                    options=[
                        ("Default", 0),
                        ("Reward Asc", 1),
                        ("Reward Desc", 2),
                    ],
                    allow_blank=False,
                )
                yield SelectionList[int](("Select ALL", 1, True), id="fields-select-all")
                with VerticalScroll(id="scroll-view2"):
                    yield SelectionList[str](*self.filter_fields, id="fields-select")
            with VerticalScroll(id="scroll-view"):
                yield Static(id="content", markup=False)

    async def on_mount(self) -> None:
        self.step_select = self.query_one("#step-select", Select)
        self.sample_select = self.query_one("#sample-select", Select)
        self.sample_sort = self.query_one("#sample-sort", Select)
        self.content_display = self.query_one("#content", Static)
        self.search_box = self.query_one("#search-box", Input)
        self.index_box = self.query_one("#index-box", Input)
        self.scroll_view = self.query_one("#scroll-view", VerticalScroll)
        self.search_status = self.query_one("#search-status", Static)
        self.fields_select = self.query_one("#fields-select", SelectionList)
        self.fields_select.border_title = "Field Filter"

        if self.data:
            self.step_select.set_options([(f"step: {i + 1}", i) for i in range(self.step_num)])
            self.sample_select.set_options([(f"sample: {i + 1}", i) for i in range(self.sample_num)])
            self.step_select.focus()
            await self.update_content()

    def update_result_options(self, offset: int = 0, sort_desc: Optional[bool] = None):
        """Update sample options based on current step and sorting."""
        options = []
        if isinstance(self.selected_step_index, int) and self.selected_step_index < len(self.data):
            if self.sample_num is None or sort_desc is not None:
                samples = self.data[self.selected_step_index].get("samples", [])
                if not samples:
                    self.selected_sample_index = offset
                    return
                if sort_desc is not None:
                    samples = sorted(
                        samples,
                        key=lambda x: float(x.get("reward", 0) if x.get("reward", "0") != "None" else 0),
                        reverse=sort_desc,
                    )
                options = [(f"sample: {r[INDEX_KEY] + 1}", r[INDEX_KEY]) for r in samples]
                self.sample_select.set_options(options)
                self.sample_num = len(samples)

            if sort_desc is not None and options:
                self.selected_sample_index = options[0][1]
            else:
                self.selected_sample_index = offset

    async def update_content(self, search_keyword: Optional[str] = None):
        """Update the content display with current sample."""
        content = ""
        try:
            samples = self.data[self.selected_step_index].get("samples", [])
            content_dict_full = samples[self.selected_sample_index]

            # Dynamically add new fields
            self._update_fields_select(content_dict_full.keys())

            # Apply field filter
            content_dict = {k: v for k, v in content_dict_full.items() if k in self.fields_select.selected}

            if self.render_table:
                content = Table("Key", "Value", show_lines=True)
                for k, v in content_dict.items():
                    v_str = str(v)
                    content.add_row(k, self.highlighter(highlight_keyword(v_str, search_keyword)))
            else:
                text = Text()
                for k, v in content_dict.items():
                    s = center_word_with_equals(k, 64) + f"\n{v}\n"
                    text.append(highlight_keyword(s, search_keyword))
                content = self.highlighter(text)
        except KeyError:
            content = f"Loading data... Progress: {len(self.data)}/{self.step_num} steps"
        except Exception:
            content = self.highlighter(traceback.format_exc())

        self.content_display.update(content)

    def _update_fields_select(self, keys):
        """Add new fields to the selection list dynamically."""
        if not hasattr(self, "fields_select"):
            return
        for k in keys:
            if k not in self._field_set:
                self._field_set.add(k)
                try:
                    self.fields_select.add_option(k, k, selected=True)
                except Exception:
                    self.fields_select.add_option((k, k, True))

    @on(Input.Submitted, "#index-box")
    async def on_index_submitted(self, event: Input.Submitted) -> None:
        """Jump to a specific sample index."""
        try:
            idx = int(event.value.strip())
            samples = self.data[self.selected_step_index].get("samples", [])
            if 0 <= idx < len(samples):
                self.selected_sample_index = idx
                self.sample_select.value = idx
                await self._clear_search()
                await self.update_content()
            else:
                self.search_status.update(Text(f"Index {idx} out of range (0-{len(samples)-1})", style="bold red"))
        except ValueError:
            self.search_status.update(Text("Invalid index", style="bold red"))

    @on(Select.Changed, "#step-select")
    async def step_changed(self, event):
        self.selected_step_index = event.value
        self.update_result_options()
        await self.update_content()

    @on(Select.Changed, "#sample-select")
    async def sample_changed(self, event):
        self.selected_sample_index = event.value
        await self._clear_search()
        await self.update_content()

    @on(Select.Changed, "#sample-sort")
    async def sort_changed(self, event):
        v = event.value
        self.update_result_options(sort_desc=None if v == 0 else False if v == 1 else True)
        await self.update_content()

    @on(SelectionList.SelectedChanged, "#fields-select")
    async def fields_changed(self, event):
        await self.update_content()

    @on(SelectionList.SelectedChanged, "#fields-select-all")
    async def fields_all_changed(self, event):
        s = self.query_one("#fields-select-all", SelectionList)
        if s.selected:
            self.fields_select.select_all()
        else:
            self.fields_select.deselect_all()

    def action_focus_previous(self):
        self.screen.focus_previous()

    def action_focus_next(self):
        self.screen.focus_next()

    async def action_next_step(self) -> None:
        self.selected_step_index = (self.selected_step_index + 1) % self.step_num
        self.step_select.value = self.selected_step_index
        self.update_result_options()
        await self.update_content()

    async def action_next_sample(self) -> None:
        self.selected_sample_index = (self.selected_sample_index + 1) % self.sample_num if self.sample_num else 0
        self.sample_select.value = self.selected_sample_index
        await self._clear_search()
        await self.update_content()

    async def action_previous_step(self) -> None:
        self.selected_step_index = (self.selected_step_index - 1) % self.step_num
        self.step_select.value = self.selected_step_index
        self.update_result_options()
        await self.update_content()

    async def action_previous_sample(self) -> None:
        self.selected_sample_index = (self.selected_sample_index - 1) % self.sample_num if self.sample_num else 0
        self.sample_select.value = self.selected_sample_index
        await self._clear_search()
        await self.update_content()

    async def action_switch_render(self):
        self.render_table = not self.render_table
        await self.update_content()

    def action_toggle_search(self) -> None:
        self.search_box.focus()

    async def action_cancel_search(self) -> None:
        self.search_box.value = ""
        await self._clear_search()
        await self.update_content()

    async def _clear_search(self):
        self.matches = []
        self.search_status.update("")
        self.current_match_index = 0

    @on(Input.Submitted, "#search-box")
    async def on_search_submitted(self, event: Input.Submitted) -> None:
        self.matches = []
        self.current_match_index = 0
        if event.value:
            await self.update_content(event.value)
            renderable = self.content_display.render()
            if isinstance(renderable, Table):
                return

            assert isinstance(renderable, Text)
            console = self.content_display._console
            lines = renderable.wrap(console, self.scroll_view.container_size.width)
            line_idx_recorded = set()
            for line_idx, line in enumerate(lines):
                if line_idx in line_idx_recorded:
                    continue
                if event.value in line:
                    self.matches.append({"line": line_idx, "word": event.value})
                    line_idx_recorded.add(line_idx)
            self.scroll_view.focus()
            await self.action_next_search()

    async def action_next_search(self) -> None:
        if not self.matches or self.current_match_index >= len(self.matches):
            return
        target_line = self.matches[self.current_match_index]["line"]
        self.scroll_view.scroll_to(x=0, y=target_line, animate=False)
        self.current_match_index = (self.current_match_index + 1) % len(self.matches)
        self.search_status.update(
            Text(f"Match: {self.current_match_index + 1}/{len(self.matches)}", style="bold on #8f51b5")
        )

    def action_page_up(self):
        self.scroll_view.scroll_page_up(animate=False)

    def action_page_down(self):
        self.scroll_view.scroll_page_down(animate=False)

    def action_page_home(self):
        self.scroll_view.scroll_home(animate=False)

    def action_page_end(self):
        self.scroll_view.scroll_end(animate=False)


async def _run(path: Path):
    """Main async entry point."""
    assert path.exists(), f"Path does not exist: {path}"

    # Find all .pt files
    paths = sorted(path.glob(f"*{FILE_SUFFIX}"), key=lambda x: x.stem)
    if not paths:
        raise ValueError(f"No .pt files found in {path}")

    print(f"Found {len(paths)} rollout files")

    pbar = ProgressBar(total=len(paths), name="Loading Progress")
    data = {}

    # Load first file synchronously for immediate display
    first_samples = load_pt_file(paths[0])
    data[0] = {"samples": first_samples, "path": str(paths[0])}
    print(f"Loaded: {paths[0].name} ({len(first_samples)} samples)")
    pbar.advance(1)

    # Create app and load remaining files in background
    app = RolloutViewer(step_num=len(paths), data=data, pbar=pbar)

    if len(paths) > 1:
        # Load remaining files asynchronously
        async def load_remaining():
            for i, p in enumerate(paths[1:], start=1):
                samples = await asyncio.get_event_loop().run_in_executor(None, load_pt_file, p)
                data[i] = {"samples": samples, "path": str(p)}
                print(f"Loaded: {p.name} ({len(samples)} samples)")
                pbar.advance(1)

        await asyncio.gather(load_remaining(), app.run_async())
    else:
        await app.run_async()


cli = typer.Typer(help="Interactive TUI for viewing slime rollout data")


@cli.command()
def run(
    rollout_dir: Annotated[Path, typer.Argument(help="Directory containing rollout .pt files")],
):
    """Launch the rollout viewer TUI."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_run(rollout_dir))


if __name__ == "__main__":
    cli()
