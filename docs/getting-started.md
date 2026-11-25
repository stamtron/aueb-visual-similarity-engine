# Getting started (Windows, macOS, VS Code)

This page is for you if you have not used a terminal much, or have never heard of `uv`.
It takes you from an empty laptop to a passing test run and your first image search,
step by step. Allow about 30 minutes, most of it waiting for downloads.

**What you will install.** Three free tools. You do **not** need to install Python
yourself: `uv` downloads the right version for this project.

| tool | what it is for |
|---|---|
| [VS Code](https://code.visualstudio.com/) | the editor, with a built-in terminal |
| [Git](https://git-scm.com/) | downloads the code (optional: you can download a ZIP instead) |
| [uv](https://docs.astral.sh/uv/) | installs Python and every library the project needs, in one command |

**What you need.** About 3 GB of free disk space and a normal internet connection. No
GPU, no account, no administrator rights for uv.

## 1. Install the tools

### Windows

1. **VS Code.** Download the installer from <https://code.visualstudio.com/> and run it.
   Keep the ticked option *Add to PATH*.
2. **Git.** Download *Git for Windows* from <https://git-scm.com/download/win> and run
   it. The default answer to every question is fine.
3. **uv.** Open **PowerShell** (press the Windows key, type `powershell`, press Enter)
   and paste this line:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   If your laptop has `winget` (most Windows 10 and 11 machines do), this works too:
   `winget install --id=astral-sh.uv -e`.

4. **Close every PowerShell and VS Code window**, then open them again. New programs
   are only found by windows opened *after* the install.

### macOS

1. **VS Code.** Download it from <https://code.visualstudio.com/>, unzip, and drag
   *Visual Studio Code* into *Applications*.
2. **Git.** Open *Terminal* (Cmd + Space, type `terminal`) and run
   `xcode-select --install`. If it says the tools are already installed, you are done.
3. **uv.** In the same Terminal:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

4. Close and reopen Terminal and VS Code.

### Check it worked

In a new terminal, each of these should print a version number:

```bash
git --version
uv --version
```

## 2. Get the code

Pick a simple folder for your projects, for example `C:\code` on Windows or
`~/code` on macOS. On Windows, avoid folders synced by **OneDrive** (often
`Documents` and `Desktop`): syncing thousands of library files is slow and can lock
them mid-install.

**With Git, from VS Code:** open VS Code, press **Ctrl + Shift + P** (macOS:
**Cmd + Shift + P**), type `Git: Clone`, press Enter, and paste:

```text
https://github.com/stamtron/aueb-visual-similarity-engine.git
```

Choose your projects folder, then click **Open** when VS Code asks.

**Without Git:** on the GitHub page, click the green **Code** button, then
**Download ZIP**. Unzip it into your projects folder, then in VS Code use
**File → Open Folder…** and select the unzipped folder.

If VS Code asks *"Do you trust the authors of the files in this folder?"*, choose
**Yes**. It may also offer to install the **recommended extensions** (Python and Ruff):
accept them.

## 3. Open the terminal inside VS Code

Menu **Terminal → New Terminal** (shortcut **Ctrl + `**, the key above Tab). A panel
opens at the bottom, already *inside the project folder*. On Windows it is PowerShell;
on macOS it is zsh. **Every command in this repo works in both.**

Type commands there and press Enter. Copy and paste works; on Windows you can also
right-click to paste.

## 4. Install and test

```bash
uv sync
```

The first time, this downloads Python 3.12, PyTorch and the other libraries (about
1 GB, 2–10 minutes). It creates a hidden `.venv` folder in the project: everything
goes there, nothing is installed system-wide. To start over, delete `.venv` and run
`uv sync` again.

Then run the tests:

```bash
uv run pytest -q
```

You should see something like `29 passed` after a few seconds. The tests use a fake
model, so they need no internet and no model download. **If this passes, your setup
works.**

You can also run the tests from VS Code's **Testing** panel (the flask icon in the left
bar) and click ▶.

> **Why `uv run`?** It runs a command with the project's Python and libraries, without
> you having to "activate" anything. Put it in front of every command in the docs:
> `uv run vsim …`, `uv run python …`.

## 5. Your first search, with no account

The quickest data is a public set of 1,000 small product photos from Hugging Face
(details in [data.md](data.md)). Run these one at a time:

```bash
uv sync --extra data
uvx --from huggingface_hub hf download ashraq/fashion-product-images-small --repo-type dataset --local-dir data/raw/fashion-small
uv run python scripts/download_fashion.py
```

You now have `data/images/` with 1,000 photos. Build an index and search it:

```bash
uv run vsim index data/images --out index.npz
uv run vsim search data/images/<an id>.jpg -k 5 --index index.npz
```

Replace `<an id>` with any file name you see in `data/images/` (type the first digits and
press Tab to complete it). The first `index` downloads the CLIP model (about 350 MB,
once). Any image path or `https://` URL works as a query. Open the result IDs in VS Code's file explorer
(`data/images/<id>.jpg`) to see what it found.

Then try clustering:

```bash
uv run vsim cluster data/images
uv run vsim clusters ds_...          # paste the id the previous command printed
```

For the H&M reference dataset (larger photos, needs a free Kaggle account), follow
[data.md](data.md). The full command list is in [usage.md](usage.md).

## Troubleshooting

| you see | what to do |
|---|---|
| `uv : The term 'uv' is not recognized…` / `command not found: uv` | Close **all** VS Code and terminal windows and reopen them. If it persists, rerun the install command from step 1. |
| `running scripts is disabled on this system` | Use the exact install line from step 1 (it includes `-ExecutionPolicy ByPass`). You never need to activate `.venv` yourself: use `uv run`. |
| `vsim : The term 'vsim' is not recognized` | Put `uv run` in front: `uv run vsim …`. |
| downloads time out (slow or university Wi-Fi) | Run the same command again; finished downloads are kept. In PowerShell, `$env:UV_HTTP_TIMEOUT=300` (macOS: `export UV_HTTP_TIMEOUT=300`) gives uv more time. |
| `Access is denied` / file in use, during `uv sync` on Windows | Something is holding the files: OneDrive, an antivirus scan or a second terminal. Move the project out of OneDrive (step 2), close other terminals, and retry. |
| VS Code shows red squiggles under every `import` | Press Ctrl + Shift + P → `Python: Select Interpreter` → pick the one in `.venv`. Run `uv sync` first if it is not listed. |
| `No image files found in …` | Check the path. Run commands from the project folder (the VS Code terminal starts there) and look for `data/images/` in the file explorer. |
| `403 Forbidden` from Kaggle | Accept the competition rules in the browser first ([data.md](data.md)). |
| Linux only: `uv sync` downloads several GB | PyTorch for Linux includes GPU libraries. It is expected, and it still runs on the CPU. |

Still stuck? Copy the **whole** error message, not just the last line, when you ask
for help.
