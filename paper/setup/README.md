# LaTeX Setup with MiKTeX and VS Code

This README documents the complete setup process for creating IEEE-formatted LaTeX documents using MiKTeX and Visual Studio Code on Windows.

---

## Prerequisites

* **Operating System:** Windows 10/11 (macOS/Linux users can substitute TeX Live for MiKTeX).
* **Internet connection** for downloading installers and packages.

---

## 1. Install MiKTeX

1. Download the **MiKTeX Basic Installer** from [https://miktex.org/download](https://miktex.org/download).
2. Run the installer:

   * Choose **Install for: Just for me** or **All users**.
   * Ensure **“Install missing packages on-the-fly”** is set to **Yes**.
3. Open **MiKTeX Console** → **Updates** → **Check for updates** → **Update now**.
4. Open **MiKTeX Console** → **Packages** → search for **ieeetran** tick it, and click + Install (this ensures you have the official IEEEtran.cls and bibliography style files).

---

## 2. Install Strawberry Perl

LaTeX Workshop uses `latexmk`, which requires Perl.

1. Download **Strawberry Perl** (64-bit) from [https://strawberryperl.com/](https://strawberryperl.com/).
2. Run the installer (it auto-adds Perl to your system `PATH`).

---

## 3. Ensure `latexmk` is available

1. Open **MiKTeX Console** → **Packages**, search for `latexmk`, and install it if missing.
2. Verify in a terminal:

   ```bash
   latexmk --version
   ```

---

## 4. Install VS Code and Extensions

1. Download VS Code from [https://code.visualstudio.com/](https://code.visualstudio.com/) and install.
2. In VS Code, open **Extensions** (Ctrl+Shift+X) and install **LaTeX Workshop**:

   ```text
   ext install latex-workshop
   ```
3. (Optional) Install **LaTeX Utilities** and **Spell Right**.

---

## 5. Verify the Toolchain

In a VS Code integrated terminal, run:

```bash
pdflatex --version
latexmk --version
perl -v
```

All commands should execute without errors.

---

## 6. Create Your LaTeX Project

1. Create a new folder for your paper and navigate into it.
2. (Optional) Initialize Git
3. Add the `.gitignore` (see provided template) to exclude build artifacts.
4. Create `main.tex` using the IEEEtran template:

```latex
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite,graphicx,amsmath,amsfonts}

\title{Your Paper Title}
\author{%
  \IEEEauthorblockN{Shreyas Rajendra Poyrekar}
  \IEEEauthorblockA{Affiliation \\ shreyas@example.com}
  \and
  \IEEEauthorblockN{Joy Rebello}
  \IEEEauthorblockA{Affiliation \\ joy@example.com}
}

\begin{document}
\maketitle

\begin{abstract}
Your abstract goes here.
\end{abstract}
\begin{IEEEkeywords}
LaTeX, IEEEtran, MiKTeX
\end{IEEEkeywords}

\section{Introduction}
Start writing your introduction...

\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}
````

---

## 7. Configure VS Code Build Settings

Create `.vscode/settings.json` in the project root:

```jsonc
{
  "latex-workshop.latex.outDir": "./build",
  "latex-workshop.latex.tools": [
    {
      "name": "latexmk",
      "command": "latexmk",
      "args": [
        "-pdf",
        "-interaction=nonstopmode",
        "-synctex=1",
        "-aux-directory=build",
        "-output-directory=build",
        "%DOC%"
      ]
    }
  ],
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk (build)",
      "tools": ["latexmk"]
    }
  ],
  "latex-workshop.view.pdf.viewer": "tab"
}
```

* **`outDir`** ensures all intermediate files and the final PDF go to `build/`.
* **Custom recipes** call `latexmk` with appropriate MiKTeX flags.

---

## 8. Build and Preview

* Press **Ctrl+Alt+B** (or run the **Build LaTeX project** command).
* The output PDF appears in `build/` and opens in a VS Code tab.

---

## 9. Clean Workflow

* All source files (`.tex`, `.bib`, figures) and the final PDF are tracked.
* Aux files remain in `build/` and are excluded by `.gitignore`.

---

## 10. Troubleshooting Tips

| Issue                  | Solution                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| `spawn latexmk ENOENT` | Install `latexmk` via MiKTeX Console and ensure Perl and MiKTeX bin paths are in `PATH`. |
| Missing packages       | Open MiKTeX Console → Settings → set “Install missing packages” to **Yes**.              |
| SyncTeX not working    | Confirm `-synctex=1` is present in your recipe args.                                     |

---

Happy typesetting with a clean, organized workflow! Feel free to customize further for your needs.
