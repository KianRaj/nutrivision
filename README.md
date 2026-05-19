# NutriVision — static landing page (GitHub Pages)

This folder is a **pure-static** snapshot of the NutriVision landing page —
HTML + CSS + JS only, no Python/Flask backend required. It's designed to be
served by **GitHub Pages** so you get a permanent, free, publicly accessible
URL like:

```
https://<your-username>.github.io/<repo-name>/
```

## What's here

```
docs/
├── index.html      # the landing page (results, gallery, architecture, optional demo)
├── style.css       # academic / forest / dark themes
├── app.js          # theme cycle + optional live-demo POSTs to a backend
├── img/            # architecture diagram + sample food photos + triptych
└── README.md       # this file
```

The live demo section in `index.html` only works if `API_URL` in `app.js` is
set to a reachable backend (your Flask server exposed through Cloudflare
Tunnel, ngrok, etc.). When `API_URL` is empty, the demo is auto-disabled with
a friendly "live demo offline" message — the rest of the page works fine.

## Deploy to GitHub Pages — step by step

### 1. Create the repo (one time)

If you don't already have a repo for NutriVision:

```bash
# in any new empty directory:
cd /tmp/nutrivision-site             # or any local working dir
git init
git branch -M main

# copy the docs/ folder over
cp -r /media/nas_mount/research3/aman_kr/midas/Om_IGSM_v2/docs/* .
```

Then create the repo on GitHub (web UI is fine), and:

```bash
git remote add origin https://github.com/<YOUR-USERNAME>/nutrivision.git
git add .
git commit -m "Initial NutriVision landing page"
git push -u origin main
```

### 2. Enable GitHub Pages

In the GitHub repo:
**Settings → Pages → Source → "Deploy from a branch" → Branch: `main`, Folder: `/ (root)` → Save.**

Wait 1–2 minutes; the URL appears at the top of the Pages settings page,
something like `https://<your-username>.github.io/nutrivision/`.

### 3. (Optional) Use the existing repo's `docs/` folder instead

If your Om_IGSM_v2 directory is already a git repo on GitHub, you can simply
serve from `/docs` without copying anywhere:

**Settings → Pages → Branch: `main`, Folder: `/docs` → Save.**

The URL becomes `https://<you>.github.io/<repo-name>/`.

## Update the live site (push changes from the institute server to GitHub)

Whenever the `docs/` folder on the institute server changes
(new results, new images, layout tweaks, etc.), follow this loop to push
the updates so they appear at `https://kianraj.github.io/nutrivision/`.

### A. Make sure your local repo folder exists

The Pages repo lives at `C:\Users\LENOVO\Desktop\nutrivision-site\` on the
laptop. If you've moved it, find it with:

```cmd
where /r C:\Users\LENOVO index.html
```

### B. Pull the updated files from the institute server

> ⚠️ **Important** — use the right shell. The commands below are written
> for **Command Prompt** (`cmd.exe`). If you open **PowerShell** or
> **Windows Terminal** instead, replace any `%VAR%` with `$env:VAR`.

Replace `<institute-server-host>` with your actual SSH host (the same
one you SSH into normally, e.g. `bhandari.iiitd.ac.in`).

**Command Prompt (cmd.exe):**

```cmd
cd C:\Users\LENOVO\Desktop\nutrivision-site

scp -r aman24012@<institute-server-host>:/media/nas_mount/research3/aman_kr/midas/Om_IGSM_v2/docs/* .
```

**PowerShell / Windows Terminal:**

```powershell
cd $env:USERPROFILE\Desktop\nutrivision-site

scp -r aman24012@<institute-server-host>:/media/nas_mount/research3/aman_kr/midas/Om_IGSM_v2/docs/* .
```

**Mac/Linux:**

```bash
cd ~/Desktop/nutrivision-site

scp -r aman24012@<institute-server-host>:/media/nas_mount/research3/aman_kr/midas/Om_IGSM_v2/docs/* .
```

When `scp` prompts for the password, enter your **institute-server**
password (not your GitHub one).

### C. Commit + push to GitHub

```cmd
git add .
git status
git commit -m "Update landing page from institute server"
git push
```

When `git push` prompts for credentials:
* Username: `KianRaj`
* Password: your **Personal Access Token** (the `ghp_…` string from
  <https://github.com/settings/tokens>) — *not* your GitHub login password.

### D. Verify the deploy

1. GitHub Actions builds the Pages site automatically. Watch progress at
   <https://github.com/KianRaj/nutrivision/actions> — wait for the green
   check (~60 seconds).
2. Hard-refresh the live URL to bust the browser cache:
   * **Chrome/Edge/Firefox on Windows/Linux:** `Ctrl + Shift + R`
   * **Safari/Chrome on Mac:** `Cmd + Shift + R`
3. <https://kianraj.github.io/nutrivision/> should now show the updates.

### E. Common errors and fixes

| Error message | Cause | Fix |
|---|---|---|
| `cd: $env:USERPROFILE\Desktop\nutrivision-site: No such file or directory` | You're in cmd.exe but using PowerShell syntax | Use `cd C:\Users\LENOVO\Desktop\nutrivision-site` (literal path) in cmd, or switch to PowerShell |
| `fatal: not a git repository` | You're not inside the cloned Pages repo | `cd` into `nutrivision-site` first; use `where /r C:\Users\LENOVO index.html` to find it |
| `Permission denied (publickey,password)` on `scp` | Wrong username or unreachable host | Replace `aman24012` and `<institute-server-host>` with your actual SSH login |
| `remote: Repository not found` on `git push` | Repo URL is wrong or token doesn't have access | Check `git remote -v`; verify the token has `repo` scope |
| `Authentication failed` on `git push` | Used password instead of Personal Access Token | Generate a new token at <https://github.com/settings/tokens/new> with `repo` scope; use it as the password |
| Page still shows old content after push | Browser cache | Hard-refresh (`Ctrl + Shift + R`). If still stale, check Actions tab — the build may still be running |

## Connect the live demo to your backend

The static page on Pages can call your Flask backend if it's reachable. The
cleanest free path is **Cloudflare Tunnel** (no sudo, free, HTTPS):

```bash
# 1. Make sure your Flask app is running locally on :5050
cd /media/nas_mount/research3/aman_kr/midas/Om_IGSM_v2
PORT=5050 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  CHECKPOINT=/media/nas_mount/research3/aman_kr/midas/Om_IGSM/checkpoints/om_igsm_run1/best_model.pt \
  DEPTH_FORMAT=v1 EXPECTED_PMAE=13.52 \
  nohup /media/data_dump/conda/miniconda3/envs/depth-pro/bin/python3.9 \
        -m demo.flask_app.server > demo/logs/app.log 2>&1 &
echo $! > demo/logs/app.pid

# 2. Run the cloudflared quick tunnel
mkdir -p ~/logs
nohup ~/bin/cloudflared tunnel --url http://localhost:5050 \
      > ~/logs/cloudflared.log 2>&1 &
sleep 6
grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' ~/logs/cloudflared.log | head -1
# → prints something like https://orange-pancake-22.trycloudflare.com
```

Paste that URL into `docs/app.js`:

```js
const API_URL = "https://orange-pancake-22.trycloudflare.com";
```

Commit and push. GitHub Pages auto-redeploys in ~1 minute, and the "Try it"
section now POSTs to your backend.

### Backend CORS

You need to allow the GitHub Pages origin to call your Flask backend. Quick
way — install `flask-cors` and patch `flask_app/server.py`:

```bash
pip install flask-cors
```

Then in `flask_app/server.py`, near the top of `create_app()`:

```python
from flask_cors import CORS
app = Flask(...)
CORS(app, resources={r"/api/*": {"origins": "*"}})
```

(For production, scope `origins` to your Pages URL.)

## Tradeoffs to be aware of

| Aspect | Static Pages site | Live Flask demo (via tunnel) |
|---|---|---|
| URL stable forever | ✅ | ❌ (trycloudflare URL changes on each restart) |
| Works when your server is off | ✅ (page still loads, demo shows "offline") | ❌ (demo errors) |
| Free | ✅ forever | ✅ if your server is already running |
| Custom domain | Free via Cloudflare Pages or paid via GitHub Pages | needs a named Cloudflare tunnel + a domain |

For a permanent demo URL that survives your server reboots, see the named-tunnel
+ custom-domain instructions in the main `demo/README.md`, or migrate the
backend to Hugging Face Spaces (CPU tier — free but slow inference).
