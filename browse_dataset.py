#!/usr/bin/env python3
"""
Interactive HuggingFace dataset browser for the command line.

Usage:
    python browse_dataset.py <dataset_name_or_path> [--split train]
    
    # Examples:
    python browse_dataset.py allenai/tulu-3-sft-mixture --split train --num 1000
    python browse_dataset.py ./local_dataset
    python browse_dataset.py output.jsonl

Controls:
    j/↓      - Next row
    k/↑      - Previous row
    h/←      - Previous column
    l/→      - Next column
    Enter/e  - Expand/collapse current cell
    g        - Go to specific row
    /        - Search in current column
    n        - Next search result
    N        - Previous search result
    c        - List all columns
    Home     - First row
    End      - Last row
    PageUp   - Jump 10 rows up
    PageDown - Jump 10 rows down
    q/Esc    - Quit
"""

import argparse
import curses
import json
import os
import sys
import textwrap


def load_dataset_smart(path: str, split: str = "train", num_samples: int | None = None):
    """Load dataset from various sources."""
    from datasets import load_dataset, load_from_disk
    
    # Check if it's a local JSONL file
    if path.endswith(".jsonl") or path.endswith(".json"):
        if path.endswith(".jsonl"):
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
                        if num_samples and len(data) >= num_samples:
                            break
            return data, list(data[0].keys()) if data else []
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                if num_samples:
                    data = data[:num_samples]
                return data, list(data[0].keys()) if data else []
            return [data], list(data.keys())
    
    # Check if it's a local directory (saved dataset)
    if os.path.isdir(path):
        dataset = load_from_disk(path)
        if hasattr(dataset, 'keys'):  # DatasetDict
            dataset = dataset[split] if split in dataset else dataset[list(dataset.keys())[0]]
    else:
        # Try loading from HuggingFace Hub
        try:
            dataset = load_dataset(path, split=split)
        except Exception:
            dataset = load_dataset(path)
            if hasattr(dataset, 'keys'):
                available = list(dataset.keys())
                dataset = dataset[split] if split in available else dataset[available[0]]
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    columns = dataset.column_names
    data = [dict(row) for row in dataset]
    return data, columns


def format_cell(value, max_width: int = 50, max_lines: int = 1) -> str:
    """Format a cell value for display."""
    if value is None:
        return "<null>"
    
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        text = str(value)
    
    # Replace newlines with visible markers for collapsed view
    if max_lines == 1:
        text = text.replace("\n", "↵ ")
        if len(text) > max_width:
            return text[:max_width - 3] + "..."
        return text
    else:
        return text


class DatasetBrowser:
    def __init__(self, data: list[dict], columns: list[str], dataset_name: str):
        self.data = data
        self.columns = columns
        self.dataset_name = dataset_name
        self.current_row = 0
        self.current_col = 0
        self.expanded = False
        self.scroll_offset = 0
        self.search_term = ""
        self.search_results = []
        self.search_idx = 0
        self.message = ""
        
    def get_cell_value(self, row: int, col: int) -> any:
        """Get value at row, column."""
        if 0 <= row < len(self.data) and 0 <= col < len(self.columns):
            return self.data[row].get(self.columns[col])
        return None
    
    def search(self, term: str):
        """Search for term in current column."""
        self.search_term = term.lower()
        self.search_results = []
        col_name = self.columns[self.current_col]
        
        for i, row in enumerate(self.data):
            val = str(row.get(col_name, "")).lower()
            if self.search_term in val:
                self.search_results.append(i)
        
        if self.search_results:
            self.search_idx = 0
            self.current_row = self.search_results[0]
            self.message = f"Found {len(self.search_results)} matches"
        else:
            self.message = "No matches found"
    
    def next_search_result(self):
        """Go to next search result."""
        if not self.search_results:
            self.message = "No search results"
            return
        self.search_idx = (self.search_idx + 1) % len(self.search_results)
        self.current_row = self.search_results[self.search_idx]
        self.message = f"Match {self.search_idx + 1}/{len(self.search_results)}"
    
    def prev_search_result(self):
        """Go to previous search result."""
        if not self.search_results:
            self.message = "No search results"
            return
        self.search_idx = (self.search_idx - 1) % len(self.search_results)
        self.current_row = self.search_results[self.search_idx]
        self.message = f"Match {self.search_idx + 1}/{len(self.search_results)}"

    def run(self, stdscr):
        """Main curses loop."""
        curses.curs_set(0)
        curses.use_default_colors()
        
        # Initialize color pairs
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)   # Header
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_YELLOW) # Selected
        curses.init_pair(3, curses.COLOR_GREEN, -1)                  # Column name
        curses.init_pair(4, curses.COLOR_CYAN, -1)                   # Info
        curses.init_pair(5, curses.COLOR_RED, -1)                    # Search highlight
        
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Header
            header = f" {self.dataset_name} | Row {self.current_row + 1}/{len(self.data)} | Col: {self.columns[self.current_col]} ({self.current_col + 1}/{len(self.columns)}) "
            header = header[:width-1]
            stdscr.addstr(0, 0, header.ljust(width-1), curses.color_pair(1) | curses.A_BOLD)
            
            # Controls hint
            controls = " j/k:row h/l:col e:expand g:goto /:search q:quit "
            stdscr.addstr(1, 0, controls[:width-1], curses.color_pair(4))
            
            if self.expanded:
                # Expanded view - show full cell content
                self.draw_expanded_view(stdscr, height, width)
            else:
                # Table view
                self.draw_table_view(stdscr, height, width)
            
            # Message line
            if self.message:
                stdscr.addstr(height - 1, 0, self.message[:width-1], curses.color_pair(5))
            
            stdscr.refresh()
            
            # Handle input
            key = stdscr.getch()
            self.message = ""
            
            if key in (ord('q'), 27):  # q or Esc
                break
            elif key in (ord('j'), curses.KEY_DOWN):
                self.current_row = min(self.current_row + 1, len(self.data) - 1)
                self.scroll_offset = 0
            elif key in (ord('k'), curses.KEY_UP):
                self.current_row = max(self.current_row - 1, 0)
                self.scroll_offset = 0
            elif key in (ord('l'), curses.KEY_RIGHT):
                self.current_col = min(self.current_col + 1, len(self.columns) - 1)
                self.scroll_offset = 0
            elif key in (ord('h'), curses.KEY_LEFT):
                self.current_col = max(self.current_col - 1, 0)
                self.scroll_offset = 0
            elif key in (ord('e'), ord('\n'), curses.KEY_ENTER):
                self.expanded = not self.expanded
                self.scroll_offset = 0
            elif key == ord('g'):
                self.goto_row(stdscr)
            elif key == ord('/'):
                self.do_search(stdscr)
            elif key == ord('n'):
                self.next_search_result()
            elif key == ord('N'):
                self.prev_search_result()
            elif key == ord('c'):
                self.show_columns(stdscr)
            elif key == curses.KEY_HOME:
                self.current_row = 0
                self.scroll_offset = 0
            elif key == curses.KEY_END:
                self.current_row = len(self.data) - 1
                self.scroll_offset = 0
            elif key == curses.KEY_PPAGE:  # Page Up
                self.current_row = max(self.current_row - 10, 0)
                self.scroll_offset = 0
            elif key == curses.KEY_NPAGE:  # Page Down
                self.current_row = min(self.current_row + 10, len(self.data) - 1)
                self.scroll_offset = 0
            # Scroll in expanded view
            elif self.expanded and key == ord('J'):
                self.scroll_offset += 1
            elif self.expanded and key == ord('K'):
                self.scroll_offset = max(0, self.scroll_offset - 1)

    def draw_table_view(self, stdscr, height: int, width: int):
        """Draw the table view."""
        start_y = 3
        available_height = height - start_y - 2
        
        # Calculate visible columns
        col_width = max(20, (width - 5) // 3)
        
        # Show a few columns around current
        visible_cols = []
        start_col = max(0, self.current_col - 1)
        for i in range(start_col, min(len(self.columns), start_col + 3)):
            visible_cols.append(i)
        
        # Column headers
        x = 0
        for col_idx in visible_cols:
            col_name = self.columns[col_idx]
            if col_idx == self.current_col:
                attr = curses.color_pair(3) | curses.A_BOLD | curses.A_UNDERLINE
            else:
                attr = curses.color_pair(3)
            header_text = col_name[:col_width-1].ljust(col_width)
            try:
                stdscr.addstr(start_y, x, header_text, attr)
            except curses.error:
                pass
            x += col_width + 1
        
        # Calculate row window
        half_height = available_height // 2
        start_row = max(0, self.current_row - half_height)
        end_row = min(len(self.data), start_row + available_height)
        
        # Adjust if near the end
        if end_row - start_row < available_height and start_row > 0:
            start_row = max(0, end_row - available_height)
        
        # Draw rows
        for y_offset, row_idx in enumerate(range(start_row, end_row)):
            y = start_y + 1 + y_offset
            if y >= height - 2:
                break
            
            x = 0
            for col_idx in visible_cols:
                value = self.get_cell_value(row_idx, col_idx)
                text = format_cell(value, col_width - 1)
                
                # Highlight current cell
                if row_idx == self.current_row and col_idx == self.current_col:
                    attr = curses.color_pair(2) | curses.A_BOLD
                elif row_idx == self.current_row:
                    attr = curses.A_REVERSE
                elif row_idx in self.search_results:
                    attr = curses.color_pair(5)
                else:
                    attr = curses.A_NORMAL
                
                cell_text = text[:col_width-1].ljust(col_width)
                try:
                    stdscr.addstr(y, x, cell_text, attr)
                except curses.error:
                    pass
                x += col_width + 1

    def draw_expanded_view(self, stdscr, height: int, width: int):
        """Draw expanded view of current cell."""
        start_y = 3
        
        # Show column name
        col_name = self.columns[self.current_col]
        stdscr.addstr(start_y, 0, f"Column: {col_name}", curses.color_pair(3) | curses.A_BOLD)
        stdscr.addstr(start_y + 1, 0, "─" * (width - 1))
        
        # Get full content
        value = self.get_cell_value(self.current_row, self.current_col)
        if value is None:
            content = "<null>"
        elif isinstance(value, (dict, list)):
            content = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            content = str(value)
        
        # Wrap and display
        lines = []
        for line in content.split("\n"):
            wrapped = textwrap.wrap(line, width - 2) or [""]
            lines.extend(wrapped)
        
        # Apply scroll offset
        visible_lines = lines[self.scroll_offset:]
        
        for i, line in enumerate(visible_lines):
            y = start_y + 2 + i
            if y >= height - 2:
                break
            try:
                stdscr.addstr(y, 0, line[:width-1])
            except curses.error:
                pass
        
        # Scroll indicator
        if len(lines) > height - start_y - 3:
            scroll_info = f" Lines {self.scroll_offset + 1}-{min(self.scroll_offset + height - start_y - 3, len(lines))}/{len(lines)} (J/K to scroll) "
            try:
                stdscr.addstr(height - 2, 0, scroll_info, curses.color_pair(4))
            except curses.error:
                pass

    def goto_row(self, stdscr):
        """Prompt user to go to specific row."""
        height, width = stdscr.getmaxyx()
        curses.echo()
        curses.curs_set(1)
        
        stdscr.addstr(height - 1, 0, "Go to row: ".ljust(width - 1))
        stdscr.refresh()
        
        try:
            input_str = stdscr.getstr(height - 1, 11, 10).decode('utf-8')
            row_num = int(input_str) - 1  # Convert to 0-indexed
            if 0 <= row_num < len(self.data):
                self.current_row = row_num
                self.message = f"Jumped to row {row_num + 1}"
            else:
                self.message = f"Invalid row (1-{len(self.data)})"
        except (ValueError, curses.error):
            self.message = "Invalid input"
        
        curses.noecho()
        curses.curs_set(0)

    def do_search(self, stdscr):
        """Prompt user for search term."""
        height, width = stdscr.getmaxyx()
        curses.echo()
        curses.curs_set(1)
        
        col_name = self.columns[self.current_col]
        prompt = f"Search in '{col_name}': "
        stdscr.addstr(height - 1, 0, prompt.ljust(width - 1))
        stdscr.refresh()
        
        try:
            input_str = stdscr.getstr(height - 1, len(prompt), width - len(prompt) - 1).decode('utf-8')
            if input_str:
                self.search(input_str)
        except curses.error:
            self.message = "Search cancelled"
        
        curses.noecho()
        curses.curs_set(0)

    def show_columns(self, stdscr):
        """Show all column names."""
        height, width = stdscr.getmaxyx()
        stdscr.clear()
        
        stdscr.addstr(0, 0, "Columns (press any key to return):", curses.color_pair(1) | curses.A_BOLD)
        
        for i, col in enumerate(self.columns):
            y = 2 + i
            if y >= height - 1:
                stdscr.addstr(height - 1, 0, f"... and {len(self.columns) - i} more", curses.color_pair(4))
                break
            marker = "→ " if i == self.current_col else "  "
            attr = curses.A_BOLD if i == self.current_col else curses.A_NORMAL
            stdscr.addstr(y, 0, f"{marker}{i + 1}. {col}"[:width-1], attr)
        
        stdscr.refresh()
        stdscr.getch()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive HuggingFace dataset browser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name (HF Hub), path (local dir), or file (.jsonl/.json)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load (default: train)",
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=None,
        help="Number of samples to load (default: all)",
    )
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset}...")
    try:
        data, columns = load_dataset_smart(args.dataset, args.split, args.num)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not data:
        print("Dataset is empty!", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(data)} rows, {len(columns)} columns")
    print("Starting browser...")
    
    browser = DatasetBrowser(data, columns, args.dataset)
    
    try:
        curses.wrapper(browser.run)
    except KeyboardInterrupt:
        pass
    
    print("Goodbye!")


if __name__ == "__main__":
    main()
