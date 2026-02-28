#!/usr/bin/env python3
"""
session.py — Session Memory Injector
A tool for maintaining project memory across AI coding sessions.

Tracks file changes, summarizes them with a local LLM, and generates
paste-ready briefings for starting new AI chat sessions.

Usage:
    session.py -n  ~/Projects/MyProject          # create new project
    session.py -ap ~/Projects/MyProject ~/some/dir       # add watched directory
    session.py -af ~/Projects/MyProject ~/some/file.py   # add watched file
    session.py -s  ~/Projects/MyProject          # start session (snapshot)
    session.py -x  ~/Projects/MyProject          # end session (diff + summarize)
    session.py -b  ~/Projects/MyProject          # print paste-ready briefing
"""

import sys
import os
import json
import shutil
import hashlib
import subprocess
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Name of the config file inside every project folder
CONFIG_FILE = "session.json"

# Name of the running memory log
MEMORY_FILE = "session_memory.md"

# Subfolder where snapshots are stored
SNAPSHOT_DIR = "snapshots"

# Ollama API endpoint (local)
OLLAMA_API = "http://localhost:11434/api/generate"

# Ollama list command
OLLAMA_LIST_CMD = ["ollama", "list"]

# Ollama install info shown to user if not found
OLLAMA_INSTALL_MSG = """
Ollama is not installed or not running.
Ollama lets you run AI models locally and privately — nothing leaves your machine.

To install:  https://ollama.com
Then run:    ollama pull qwen2:1.5b   (small, fast, good for summarizing)

Once installed, come back and run this script again.
"""


# ---------------------------------------------------------------------------
# HELPER: Print a clear section divider
# ---------------------------------------------------------------------------
def divider():
    print("\n" + "=" * 50 + "\n")


# ---------------------------------------------------------------------------
# HELPER: Ask user a yes/no question, return True for yes
# ---------------------------------------------------------------------------
def ask_yes_no(question):
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please enter y or n.")


# ---------------------------------------------------------------------------
# HELPER: Expand and resolve a path (handles ~, relative paths)
# ---------------------------------------------------------------------------
def resolve_path(path_str):
    return str(Path(path_str).expanduser().resolve())


# ---------------------------------------------------------------------------
# OLLAMA: Check if Ollama is available and return list of installed models.
# Returns empty list if Ollama is not installed or not running.
# ---------------------------------------------------------------------------
def get_ollama_models():
    """
    Calls 'ollama list' and parses the output.
    Returns a list of model name strings, or empty list if unavailable.
    """
    try:
        result = subprocess.run(
            OLLAMA_LIST_CMD,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:  # skip header row
            parts = line.split()
            if parts:
                models.append(parts[0])  # first column is model name
        return models
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


# ---------------------------------------------------------------------------
# OLLAMA: Send a prompt to the local model and return the response text.
# ---------------------------------------------------------------------------
def ollama_ask(model, prompt):
    """
    Sends a prompt to the specified Ollama model via its local HTTP API.
    Returns the response string, or an error message if something goes wrong.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_API,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        return f"[Ollama error: {e}]"


# ---------------------------------------------------------------------------
# CONFIG: Load session.json from a project folder.
# Exits with a clear error if it doesn't exist.
# ---------------------------------------------------------------------------
def load_config(project_path):
    """
    Loads and returns the session.json config dict from the given project folder.
    Exits with a friendly error if the file is missing.
    """
    config_path = os.path.join(project_path, CONFIG_FILE)
    if not os.path.exists(config_path):
        print(f"\nNo project found at: {project_path}")
        print(f"Expected config file: {config_path}")
        print(f"\nTo create a new project, run:  session.py -n {project_path}\n")
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CONFIG: Save session.json to a project folder.
# ---------------------------------------------------------------------------
def save_config(project_path, config):
    """
    Writes the config dict to session.json in the given project folder.
    """
    config_path = os.path.join(project_path, CONFIG_FILE)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


# Max file size to store full contents for diffing (100KB)
MAX_CONTENT_SIZE = 100 * 1024

# Extensions we treat as text and will diff
TEXT_EXTENSIONS = {
    ".py", ".txt", ".md", ".json", ".html", ".php", ".js",
    ".css", ".sh", ".conf", ".cfg", ".ini", ".yaml", ".yml",
    ".lua", ".c", ".h", ".rs", ".toml", ".log"
}

# Extensions we skip entirely (binary, generated, media)
SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".db", ".sqlite", ".png", ".jpg",
    ".jpeg", ".gif", ".ico", ".mp3", ".mp4", ".bin",
    ".zip", ".tar", ".gz", ".o", ".so"
}


# ---------------------------------------------------------------------------
# SNAPSHOT: Hash a single file's contents for change detection.
# ---------------------------------------------------------------------------
def hash_file(filepath):
    """
    Returns an MD5 hex digest of the file contents.
    Returns None if the file cannot be read (missing, permission error, etc).
    """
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except (OSError, IOError):
        return None


# ---------------------------------------------------------------------------
# SNAPSHOT: Read a text file's contents for diffing.
# Returns None if the file is too large, unreadable, or not a text file.
# ---------------------------------------------------------------------------
def read_text_content(filepath):
    """
    Reads and returns the text content of a file if it is small enough
    and has a known text extension. Returns None otherwise.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in TEXT_EXTENSIONS:
        return None
    try:
        size = os.path.getsize(filepath)
        if size > MAX_CONTENT_SIZE:
            return None
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except (OSError, IOError):
        return None


# ---------------------------------------------------------------------------
# SNAPSHOT: Collect all files to watch based on the config's watch list.
# Returns a dict of { filepath: {"hash": str, "content": str or None} }
# ---------------------------------------------------------------------------
def collect_files(config):
    """
    Walks all watched paths from the config and returns a dict mapping
    each file path to a dict with its MD5 hash and text content (if readable).
    Directories are walked recursively.
    Skips binary files, hidden files, and snapshot directories.
    """
    file_map = {}

    for watched in config.get("watch", []):
        watched = resolve_path(watched)
        if os.path.isfile(watched):
            ext = os.path.splitext(watched)[1].lower()
            if ext not in SKIP_EXTENSIONS:
                file_map[watched] = {
                    "hash": hash_file(watched),
                    "content": read_text_content(watched)
                }
        elif os.path.isdir(watched):
            for root, dirs, files in os.walk(watched):
                # Skip hidden directories and snapshot directories
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".")
                    and d != SNAPSHOT_DIR
                ]
                for filename in files:
                    if filename.startswith("."):
                        continue
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in SKIP_EXTENSIONS:
                        continue
                    full_path = os.path.join(root, filename)
                    file_map[full_path] = {
                        "hash": hash_file(full_path),
                        "content": read_text_content(full_path)
                    }

    return file_map


# ---------------------------------------------------------------------------
# SNAPSHOT: Save a snapshot dict to the snapshots/ folder.
# ---------------------------------------------------------------------------
def save_snapshot(project_path, snapshot, label):
    """
    Saves a file-hash snapshot dict as JSON in the project's snapshots/ folder.
    label should be 'before' or 'after'.
    """
    snap_dir = os.path.join(project_path, SNAPSHOT_DIR)
    os.makedirs(snap_dir, exist_ok=True)
    snap_path = os.path.join(snap_dir, f"{label}.json")
    with open(snap_path, "w") as f:
        json.dump(snapshot, f, indent=4)


# ---------------------------------------------------------------------------
# SNAPSHOT: Load a snapshot from the snapshots/ folder.
# Returns empty dict if it doesn't exist.
# ---------------------------------------------------------------------------
def load_snapshot(project_path, label):
    """
    Loads and returns a snapshot dict from the snapshots/ folder.
    Returns an empty dict if no snapshot exists for that label.
    """
    snap_path = os.path.join(project_path, SNAPSHOT_DIR, f"{label}.json")
    if not os.path.exists(snap_path):
        return {}
    with open(snap_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# DIFF: Compare before/after snapshots and return a human-readable diff string.
# Also returns a detailed text diff for changed files where content is available.
# ---------------------------------------------------------------------------
def build_diff_report(before, after):
    """
    Compares two file snapshots (before and after a session).

    Each snapshot is a dict of { filepath: {"hash": str, "content": str|None} }

    Returns a tuple:
        summary   — short plain-English file list (new/modified/deleted)
        detailed  — actual line-by-line diffs for changed text files
    """
    import difflib

    before_paths = set(before.keys())
    after_paths = set(after.keys())

    new_files = after_paths - before_paths
    deleted_files = before_paths - after_paths
    common_files = before_paths & after_paths
    modified_files = {
        p for p in common_files
        if before[p].get("hash") != after[p].get("hash")
    }

    summary_lines = []

    if new_files:
        summary_lines.append("NEW FILES:")
        for p in sorted(new_files):
            summary_lines.append(f"  + {p}")

    if modified_files:
        summary_lines.append("MODIFIED FILES:")
        for p in sorted(modified_files):
            summary_lines.append(f"  ~ {p}")

    if deleted_files:
        summary_lines.append("DELETED FILES:")
        for p in sorted(deleted_files):
            summary_lines.append(f"  - {p}")

    if not summary_lines:
        return "No file changes detected.", ""

    summary = "\n".join(summary_lines)

    # Build detailed text diffs for modified files where we have content
    detail_lines = []
    for p in sorted(modified_files):
        before_content = before[p].get("content")
        after_content = after[p].get("content")

        if before_content is None or after_content is None:
            detail_lines.append(f"\n--- {os.path.basename(p)} (binary or too large to diff) ---")
            continue

        before_lines = before_content.splitlines(keepends=True)
        after_lines = after_content.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"before/{os.path.basename(p)}",
            tofile=f"after/{os.path.basename(p)}",
            n=3  # lines of context around each change
        ))

        if diff:
            detail_lines.append(f"\n--- {os.path.basename(p)} ---")
            # Cap diff output at 100 lines to keep LLM prompt manageable
            detail_lines.extend(diff[:100])
            if len(diff) > 100:
                detail_lines.append(f"... ({len(diff) - 100} more lines not shown)\n")

    # Also note new file contents briefly
    for p in sorted(new_files):
        content = after[p].get("content")
        if content:
            preview = content[:500]
            detail_lines.append(f"\n--- NEW: {os.path.basename(p)} ---")
            detail_lines.append(preview)
            if len(content) > 500:
                detail_lines.append("... (truncated)")

    detailed = "".join(detail_lines)
    return summary, detailed


# ---------------------------------------------------------------------------
# MEMORY: Append a session summary entry to session_memory.md
# ---------------------------------------------------------------------------
def append_memory(project_path, summary_text, diff_report):
    """
    Appends a dated session summary block to the project's session_memory.md.
    Creates the file with a header if it doesn't exist yet.
    """
    memory_path = os.path.join(project_path, MEMORY_FILE)
    config = load_config(project_path)
    project_name = config.get("name", "Unknown Project")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Create the file with a header if this is the first entry
    if not os.path.exists(memory_path):
        with open(memory_path, "w") as f:
            f.write(f"# Session Memory: {project_name}\n")
            f.write("Generated by session.py — do not edit the headers manually.\n\n")
            f.write("---\n\n")

    with open(memory_path, "a") as f:
        f.write(f"## Session: {timestamp}\n\n")
        f.write("### What Changed\n")
        f.write(diff_report + "\n\n")
        f.write("### Summary\n")
        f.write(summary_text + "\n\n")
        f.write("---\n\n")

    print(f"\nMemory updated: {memory_path}")


# ---------------------------------------------------------------------------
# COMMAND: -n  Create a new project
# ---------------------------------------------------------------------------
def cmd_new(project_path):
    """
    Creates a new project folder and config file.
    Asks the user for a project name and which Ollama model to use.
    Optionally adds a starting watch path.
    """
    project_path = resolve_path(project_path)

    divider()
    print("Creating new project...")
    print(f"Location: {project_path}")

    # Create the project folder
    os.makedirs(project_path, exist_ok=True)

    # Check if config already exists
    config_path = os.path.join(project_path, CONFIG_FILE)
    if os.path.exists(config_path):
        print(f"\nA project already exists here: {config_path}")
        if not ask_yes_no("Overwrite it?"):
            print("Cancelled.")
            sys.exit(0)

    # Ask for project name
    divider()
    print("What is this project called?")
    print("(This is just a label for your briefings — type anything.)")
    name = input("Project name: ").strip()
    if not name:
        name = os.path.basename(project_path)
        print(f"Using folder name: {name}")

    # Detect and pick a model
    divider()
    print("Checking for local AI models (Ollama)...")
    models = get_ollama_models()

    if not models:
        print(OLLAMA_INSTALL_MSG)
        sys.exit(1)

    print("Models available on this machine:\n")
    for i, model in enumerate(models, 1):
        print(f"  {i}) {model}")
    print()

    while True:
        choice = input("Pick a model number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            selected_model = models[int(choice) - 1]
            break
        print(f"  Please enter a number between 1 and {len(models)}.")

    print(f"\nModel selected: {selected_model}")

    # Build initial config
    config = {
        "name": name,
        "model": selected_model,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "watch": []
    }

    # Optionally add a starting path
    divider()
    print("Do you want to add a starting path to watch?")
    print("(You can always add more later with -ap or -af)")
    if ask_yes_no("Add a path now?"):
        start_path = input("Path to watch: ").strip()
        start_path = resolve_path(start_path)
        if os.path.exists(start_path):
            config["watch"].append(start_path)
            print(f"Added: {start_path}")
        else:
            print(f"Warning: path does not exist yet, adding anyway: {start_path}")
            config["watch"].append(start_path)

    # Save config
    save_config(project_path, config)

    # Create snapshot dir
    os.makedirs(os.path.join(project_path, SNAPSHOT_DIR), exist_ok=True)

    divider()
    print(f"Project created: {project_path}")
    print(f"Config:          {os.path.join(project_path, CONFIG_FILE)}")
    print(f"Memory log:      {os.path.join(project_path, MEMORY_FILE)}  (created on first session end)")
    print()
    print("Next steps:")
    print(f"  Add more paths:    session.py -ap {project_path} ~/some/directory")
    print(f"  Start a session:   session.py -s  {project_path}")
    print()


# ---------------------------------------------------------------------------
# COMMAND: -ap  Add a watched directory to an existing project
# ---------------------------------------------------------------------------
def cmd_add_path(project_path, new_path):
    """
    Adds a directory to the watch list in the project config.
    """
    project_path = resolve_path(project_path)
    new_path = resolve_path(new_path)

    config = load_config(project_path)

    if new_path in config["watch"]:
        print(f"\nAlready watching: {new_path}")
        sys.exit(0)

    if not os.path.exists(new_path):
        print(f"\nWarning: path does not exist yet: {new_path}")
        if not ask_yes_no("Add it anyway?"):
            print("Cancelled.")
            sys.exit(0)

    config["watch"].append(new_path)
    save_config(project_path, config)
    print(f"\nAdded directory: {new_path}")
    print(f"Project '{config['name']}' now watching {len(config['watch'])} path(s).")


# ---------------------------------------------------------------------------
# COMMAND: -af  Add a watched file to an existing project
# ---------------------------------------------------------------------------
def cmd_add_file(project_path, new_file):
    """
    Adds a specific file to the watch list in the project config.
    """
    project_path = resolve_path(project_path)
    new_file = resolve_path(new_file)

    config = load_config(project_path)

    if new_file in config["watch"]:
        print(f"\nAlready watching: {new_file}")
        sys.exit(0)

    if not os.path.exists(new_file):
        print(f"\nWarning: file does not exist yet: {new_file}")
        if not ask_yes_no("Add it anyway?"):
            print("Cancelled.")
            sys.exit(0)

    config["watch"].append(new_file)
    save_config(project_path, config)
    print(f"\nAdded file: {new_file}")
    print(f"Project '{config['name']}' now watching {len(config['watch'])} path(s).")


# ---------------------------------------------------------------------------
# COMMAND: -s  Start a session (take a before snapshot)
# ---------------------------------------------------------------------------
def cmd_start(project_path):
    """
    Takes a snapshot of all watched files before a coding session begins.
    This snapshot is used later by -x (end session) to detect changes.
    """
    project_path = resolve_path(project_path)
    config = load_config(project_path)

    print(f"\nStarting session for: {config['name']}")
    print("Taking snapshot of watched files...")

    snapshot = collect_files(config)
    save_snapshot(project_path, snapshot, "before")

    print(f"Snapshot taken: {len(snapshot)} file(s) tracked.")
    print(f"\nGo do your work. When done, run:")
    print(f"  session.py -x {project_path}")
    print()


# ---------------------------------------------------------------------------
# COMMAND: -x  End a session (diff, summarize, update memory)
# ---------------------------------------------------------------------------
def cmd_end(project_path):
    """
    Takes an after-snapshot, diffs it against the before-snapshot,
    sends the diff to the local LLM for a plain-English summary,
    and appends everything to session_memory.md.
    """
    project_path = resolve_path(project_path)
    config = load_config(project_path)

    print(f"\nEnding session for: {config['name']}")

    # Check that a before snapshot exists
    before = load_snapshot(project_path, "before")
    if not before:
        print("\nNo start snapshot found.")
        print(f"Did you run:  session.py -s {project_path}  before your session?")
        sys.exit(1)

    # Take after snapshot
    print("Taking after-snapshot...")
    after = collect_files(config)
    save_snapshot(project_path, after, "after")

    # Build diff report
    summary, detailed = build_diff_report(before, after)
    print("\nChanges detected:\n")
    print(summary)

    if summary == "No file changes detected.":
        print("\nNothing changed — memory not updated.")
        sys.exit(0)

    # Ask LLM to summarize
    print(f"\nAsking {config['model']} to summarize the changes...")
    print("(This may take a moment...)\n")

    # Use detailed diff if available, otherwise fall back to summary only
    diff_for_llm = detailed if detailed.strip() else summary

    prompt = f"""You are a developer's assistant helping maintain a project memory log.

Project: {config['name']}
Date: {datetime.now().strftime('%Y-%m-%d')}

Here are the actual changes made during this coding session:

{diff_for_llm}

Please write a short, plain-English summary (3-6 sentences) of what was worked on.
Describe what actually changed in the code or content — be specific.
Mention any functions added or modified, config values changed, bugs that appear to have been fixed.
This will be read at the start of the next session to jog memory.
"""

    summary_text = ollama_ask(config["model"], prompt)
    print("Summary from LLM:\n")
    print(summary_text)

    # Append to memory file
    append_memory(project_path, summary_text, summary)

    print(f"\nSession ended. Run this to get your next-session briefing:")
    print(f"  session.py -b {project_path}")
    print()


# ---------------------------------------------------------------------------
# COMMAND: -b  Print a paste-ready session briefing
# ---------------------------------------------------------------------------
def cmd_briefing(project_path):
    """
    Reads session_memory.md and formats a clean, paste-ready briefing
    for starting a new AI chat session.
    """
    project_path = resolve_path(project_path)
    config = load_config(project_path)
    memory_path = os.path.join(project_path, MEMORY_FILE)

    if not os.path.exists(memory_path):
        print(f"\nNo memory log found yet for: {config['name']}")
        print("Run a full session first (start with -s, end with -x).")
        sys.exit(0)

    with open(memory_path, "r") as f:
        memory_contents = f.read()

    divider()
    print("PASTE THIS INTO YOUR AI CHAT SESSION:")
    divider()
    print(f"# Project: {config['name']}")
    print(f"# Briefing generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"# Watched paths: {len(config.get('watch', []))}")
    print()
    print(memory_contents)
    divider()


# ---------------------------------------------------------------------------
# MAIN: Parse command-line arguments and dispatch to the right command
# ---------------------------------------------------------------------------
def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    flag = args[0]

    # -n  New project
    if flag == "-n":
        if len(args) < 2:
            print("Usage: session.py -n ~/Projects/MyProject")
            sys.exit(1)
        cmd_new(args[1])

    # -ap  Add watched directory
    elif flag == "-ap":
        if len(args) < 3:
            print("Usage: session.py -ap ~/Projects/MyProject ~/path/to/watch")
            sys.exit(1)
        cmd_add_path(args[1], args[2])

    # -af  Add watched file
    elif flag == "-af":
        if len(args) < 3:
            print("Usage: session.py -af ~/Projects/MyProject ~/path/to/file.py")
            sys.exit(1)
        cmd_add_file(args[1], args[2])

    # -s  Start session
    elif flag == "-s":
        if len(args) < 2:
            print("Usage: session.py -s ~/Projects/MyProject")
            sys.exit(1)
        cmd_start(args[1])

    # -x  End session
    elif flag == "-x":
        if len(args) < 2:
            print("Usage: session.py -x ~/Projects/MyProject")
            sys.exit(1)
        cmd_end(args[1])

    # -b  Briefing
    elif flag == "-b":
        if len(args) < 2:
            print("Usage: session.py -b ~/Projects/MyProject")
            sys.exit(1)
        cmd_briefing(args[1])

    else:
        print(f"Unknown flag: {flag}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
