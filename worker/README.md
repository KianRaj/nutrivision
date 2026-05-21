# NutriVision — Cloudflare Worker (Git-deployed)

A 60-line Cloudflare Worker that gives [NutriVision](https://kianraj.github.io/nutrivision/)
a permanent public URL in front of the Flask backend running on the IIIT-Delhi
research server.

## What this is

```
Browser  ──► nutrivision.aman24012.workers.dev   ◄── this Worker (Cloudflare edge)
                          │
                          ▼
              purchasing-budapest-...trycloudflare.com   (Cloudflare quick tunnel)
                          │
                          ▼
              http://localhost:5050               (Flask, institute server, GPU)
```

The Worker URL **never changes**. If the quick-tunnel URL ever rotates
(server reboot, network blip), update the `UPSTREAM` line in
[`src/worker.js`](src/worker.js) and `git push` — Cloudflare auto-redeploys
within ~30 seconds.

## Updating the upstream

```bash
# 1. Edit src/worker.js, change the UPSTREAM line
# 2. Commit and push
git add src/worker.js
git commit -m "Point worker at new tunnel URL"
git push
```

Watch the deploy at
<https://dash.cloudflare.com/?to=/:account/workers/services/view/nutrivision/production/deployments>.

## Local dev (optional)

```bash
npm install --global wrangler   # one-time
wrangler dev                    # starts a local proxy at http://localhost:8787
```

## Cost

Free. Cloudflare Workers free tier = 100,000 requests/day, plenty for a thesis demo.

## License

MIT.
