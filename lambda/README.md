# NutriVision — AWS Lambda (Container)

Self-contained Flask backend packaged as an AWS Lambda container image,
auto-deployed from this repo by GitHub Actions.

```
GitHub push  ──► Actions (Ubuntu runner)
                 │
                 ├─ docker buildx build  (~8 min)
                 │    └─ pulls model from huggingface.co/aman24012/nutrivision-om-igsm
                 ├─ docker push to ECR    (~5 min for first push, cached after)
                 └─ aws lambda update-function-code

Cloudflare Worker  ──► Lambda Function URL  ──► Flask  ──► OmIGSMNet inference
```

## What this gives you

- **Permanent public URL** like `https://<id>.lambda-url.ap-south-1.on.aws/`
- **Auto-deploy** on every push to `main`
- **Cold start**: ~30 s (PyTorch + ConvNeXt-Base + CLIP + BLIP). **Warm**: ~3-5 s.
- **Cost**: $0 within AWS Lambda's permanent 1M req + 400k GB-s/month free tier.
- **Institute server no longer needed.**

## Required secrets (one-time, in GitHub UI)

In the repo settings → Secrets and variables → Actions → New repository secret:

| Name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID`     | `AKIA…` of the `nutrivision-deployer` IAM user |
| `AWS_SECRET_ACCESS_KEY` | the secret you saved to CSV |

The workflow uses these to log in to ECR + manage Lambda + create the IAM role.

## How to update the model later

Bump the model in `aman24012/nutrivision-om-igsm` HF repo → push **any commit** to this folder (touch `Dockerfile`) → Actions rebuilds with the new checkpoint.

## Manual deploy trigger

In GitHub UI: **Actions** tab → **"Build & deploy NutriVision to AWS Lambda"** → **Run workflow**.
