# Task for Codex: publish this repo to GitHub

The project at `D:\worldcup-ml` is a git repo with all work already committed
(6 commits, clean working tree). Push it to a new PUBLIC GitHub repo on my account.

## Do this
1. `cd /d D:\worldcup-ml` (it's already a git repo — do NOT re-init or wipe history).
2. Confirm it's clean: `git status` should say nothing to commit; `git log --oneline`
   should show ~6 commits ending at "feat: live forecast ... Feynman blog".
3. Create the remote and push. Prefer the GitHub CLI if it's installed and I'm
   authenticated:
   ```
   gh repo create worldcup-ml --public --source=. --remote=origin --push
   ```
   If `gh` is not available, create an empty public repo named `worldcup-ml` on my
   GitHub via the web/API, then:
   ```
   git remote add origin https://github.com/<MY_USERNAME>/worldcup-ml.git
   git branch -M main
   git push -u origin main
   ```
4. Print the final repo URL.

## Notes
- Include everything that's committed (code in `src/`, data in `data/raw/`, the
  `outputs/` results and `predicted_bracket.svg`, `BLOG.md`, `README.md`). The data
  files are small and are meant to ship with the repo.
- Do not add, delete, or rewrite any files or commits. This is a publish-only task.
- If the branch is `master` locally, either keep it or rename to `main` before push;
  just report which branch you pushed.
- Do not commit any secrets (there are none in the repo — keep it that way).
