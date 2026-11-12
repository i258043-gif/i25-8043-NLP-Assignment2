# GitHub repository creation and upload

This guide follows the project naming and submission details used in this package.

## 1. Create the GitHub repository

1. Sign in to GitHub.
2. Select **New repository** (the `+` menu at the top-right, then **New repository**).
3. Enter this repository name exactly:

   `i25-8043-NLP-Assignment2`

4. Set visibility to **Public** if your assignment submission requires a public repository.
5. Leave **Add a README file**, **Add .gitignore**, and **Choose a license** unchecked. The project already contains these files.
6. Select **Create repository**.

The report supplied with this project lists this repository address:

https://github.com/i258043-gif/i25-8043-NLP-Assignment2

If that GitHub account is not yours, replace the username in the remote URL with your own GitHub username before pushing.

## 2. Extract the ZIP

1. Download and extract `i25-8043_Assignment2_DS-A.zip`.
2. Open the extracted `i25-8043_Assignment2_DS-A` folder.
3. Open a terminal in that folder (the folder containing `README.md`, `src/`, `models/`, and `requirements.txt`).

## 3. Upload using Git (recommended)

Install Git if it is not already installed. Then run these commands from the project folder:

```bash
git init
git add .
git status
git commit -m "Submit NLP Assignment 2 project"
git branch -M main
git remote add origin https://github.com/i258043-gif/i25-8043-NLP-Assignment2.git
git push -u origin main
```

If Git says that `origin` already exists, use:

```bash
git remote set-url origin https://github.com/i258043-gif/i25-8043-NLP-Assignment2.git
git push -u origin main
```

GitHub may ask you to authenticate in a browser or with a personal access token. Do not put a password or token in the repository or in the README.

### Optional: use the included incremental commit script

The package includes `make_commits.sh`, which creates a sequence of meaningful commits for the project stages. In Git Bash, WSL, or a Linux/macOS terminal, run:

```bash
bash make_commits.sh
git remote add origin https://github.com/i258043-gif/i25-8043-NLP-Assignment2.git
git branch -M main
git push -u origin main
```

Use either the single-commit method or the incremental script method; do not run both on the same fresh repository.

## 4. Check the uploaded repository

After pushing, refresh the GitHub repository page and verify that it contains:

- `README.md` and `report.pdf`
- `i25-8043_Assignment2_DS-A.ipynb`
- `src/`, `tools/`, `data/`, `models/`, `embeddings/`, `results/`, and `figures/`
- `cleaned.txt`, `raw.txt`, `Metadata.json`, and `requirements.txt`

Open the notebook on GitHub to confirm it renders. Download `report.pdf` from GitHub once to check that the correct final report was uploaded.

## 5. Large-file note

GitHub blocks individual regular Git files larger than 100 MB. The approximately 400 MB `embeddings/ppmi_matrix.npy` is therefore excluded by the project's `.gitignore`. It can be regenerated after installing the dependencies:

```bash
python -m src.part1 A
```

The remaining model checkpoints and result files are included in the repository. Do not remove the `.gitignore` rule for `embeddings/ppmi_matrix.npy` unless you intentionally configure Git LFS.

## 6. If you prefer GitHub's website

You can also create the repository using Step 1, then choose **Add file → Upload files** on GitHub and drag the extracted project contents into the page. Commit the upload. The Git method above is recommended because it handles nested folders and future updates more reliably.
