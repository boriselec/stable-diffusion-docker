# Spike: FLUX.2 [klein] 4B on CPU

Image generation server: [FLUX.2 [klein]
4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) on the processor with
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp). `server.py`
downloads the models on first start (5.2 GB, sha256 checked, see `MODELS`), runs
`sd-server` on localhost and forwards requests to it, one at a time.

## Run

Needs about 6 GB of disk and 6 GB of RAM.

```sh
docker run -d --name flux -p 8081:8081 \
  --read-only --cap-drop ALL --security-opt no-new-privileges --memory 7g \
  -v flux-models:/home/huggingface/.cache/huggingface \
  ghcr.io/boriselec/stable-diffusion-docker:spike-flux-2
# once `docker logs flux` says "Server started":
curl -o image.png --data 'a lighthouse at sunset, oil painting' http://localhost:8081/generate
```

## Settings

| Environment variable |  |
| --- | --- |
| `SD_DIFFUSION_MODEL`, `SD_LLM`, `SD_VAE` | other models, as `<owner>/<repo>/<file>` on Hugging Face |
| `SD_EXTRA_ARGS` | more `sd-server` arguments, override the defaults `-W 512 -H 512 --steps 4`, e.g. `-v` or `-W 768 -H 768 --steps 8` |

Pushing a `spike/**` branch publishes
`ghcr.io/boriselec/stable-diffusion-docker:<branch>` (slashes replaced by dashes).
