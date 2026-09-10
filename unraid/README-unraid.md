# TextyMcSpeechy on Unraid

<img src="textymcspeechy.png" width="96" alt="">

Train a Piper TTS voice on your Unraid server's NVIDIA GPU, from a browser.

Drop in a long recording. It gets split into clips, transcribed, and grouped by
speaker. You correct the transcripts and drop the clips you do not want, press
train, and listen to the voice as it improves. No local build, no host packages,
no scripts to install.

```
https://raw.githubusercontent.com/adman234/TextyMcSpeechy-unraid/main/unraid/textymcspeechy-unraid.xml
```

---

## Install

1. **Nvidia-Driver plugin** from Community Applications, then reboot. Confirm
   `nvidia-smi` on the Unraid terminal lists your card before going further.
2. Docker tab → **Add Container** → paste the template URL above into
   **Template** at the top.
3. Check the **Appdata** path points at an NVMe pool (see [Storage](#storage)),
   then **Apply**. The image is ~15 GB, so the first pull takes a while.
4. Click **WebUI** (port 8080). The header shows whether the GPU is working.
5. Pretrained checkpoints are a separate ~5 GB download. From the container
   console, run `tms checkpoints en-us` once.

That's it. There is no step where you build anything.

## Using it: the web UI

Click **WebUI** on the container (port 8080). The whole workflow lives there.

**1 · Import.** Drop in one long recording — an interview, a podcast episode,
two hours is fine. Press *Split & transcribe*. It decodes the audio, transcribes
it with word-level timestamps, cuts it into utterance-length clips at natural
pauses, and groups the clips by who is speaking.

**2 · Review.** This is where the quality comes from, so it is built to be fast:

| Key | Does |
|---|---|
| `J` / `K` | move down / up |
| `Space` | play the current clip |
| `E` | include / exclude |
| `Enter` | edit the transcript, `Esc` to stop |

On a podcast, the speaker chips at the top let you keep only your target speaker
in one click instead of excluding hundreds of clips by hand. The header tracks
**included minutes** as you go — that is the number that matters, and the UI
warns you below about ten.

Transcripts must match what was actually *said*. Whisper gets most of it, but
numbers, names and filler words need your eyes. This is the single biggest
factor in how good the finished voice sounds.

**3 · Train.** Press *Start training*. Live log and epoch counter in the browser,
loss curves on port 6006. Training survives closing the tab.

**4 · Listen.** Once checkpoints appear, type a sentence and hear the voice at
each one. That is how you decide it is done — the loss curve is a hint, your
ears are the judge. Finished voices land in
`<voice>_dojo/tts_voices/` in your appdata.

### Speaker grouping

Clips are grouped by clustering voice embeddings, deliberately not with
pyannote: its diarization models sit behind a Hugging Face licence acceptance,
which would turn "install the container" into "go make an account". The
trade-off is less precision on overlapping speech. If the embedding model is
unavailable for any reason the import still succeeds with every clip in one
group — you just exclude by hand.

### Model downloads

Speech recognition models (~3 GB) download on first import, and pretrained Piper
checkpoints (~5 GB) come from `tms checkpoints en-us`. Both land in **appdata,
not the image**, so they survive updates and are not part of the pull.

## Using it: the terminal

Everything above is also available from the container console, which is what the
web UI drives underneath:

```
tms doctor              Check the GPU, driver and PyTorch build. Run this first.
tms checkpoints en-us   Download pretrained voices to build on. Once, ~5GB.
tms dataset myvoice     Prepare recordings you put in DATASETS/myvoice/.
tms new myvoice         Create a training workspace.
tms train myvoice       Start (or resume) training.
tms attach              Reattach to a training session you detached from.
tms shell               A shell in the dojo, for anything not covered above.
```

A first voice, start to finish:

```bash
tms doctor
tms checkpoints en-us

# put your .wav files and metadata.csv in
#   /mnt/user/appdata/textymcspeechy/DATASETS/myvoice/
tms dataset myvoice

tms new myvoice
tms train myvoice
```

`tms train` opens a live training console. **Detach with `Ctrl-b` then `d`** and
it keeps running — you can close the browser tab. `tms attach` gets you back.
Watch loss curves at `http://<unraid-ip>:6006`.

Finished voices land in
`/mnt/user/appdata/textymcspeechy/myvoice_dojo/tts_voices/` as an `.onnx` and
`.onnx.json` pair, ready for Home Assistant's Piper add-on.

For what goes *in* a dataset, and how the training workflow behaves, upstream's
[quick start guide](../quick_start_guide.md) and
[dojo guide](../tts_dojo/TTS_dojo_guide.md) are the reference. They describe the
same scripts; you just reach them through `tms`.

---

## Requirements

### GPU

An NVIDIA GPU and the Unraid Nvidia-Driver plugin, with `--runtime=nvidia` left
in Extra Parameters.

**RTX 50-series (Blackwell)** needs a driver branch built from NVIDIA's **open
kernel modules**. Older branches build, load, and then simply don't see the card
— Unraid users hit exactly this on the 5060 Ti with driver `575.57.08`. Pick a
recent branch in the plugin.

`tms doctor` checks the whole chain — driver, CUDA, and whether the bundled
PyTorch actually has kernels for your card's compute capability. Run it before
investing time in a dataset. On a 50-series card you want `sm_120` to appear in
the arch list; the image ships a CUDA 13 torch build, so it should.

### Storage

Budget **50 GB free**. The image is ~15 GB, pretrained checkpoints are ~800 MB
each (six of them), and training writes new checkpoints every couple of epochs.

Two Unraid-specific points, both worth getting right the first time:

- Put appdata on an **NVMe pool, not the array**. Training reads the dataset
  constantly and writes in large bursts.
- Set that share to **Cache: Only**. Otherwise the mover will start relocating an
  active training run onto spinning disks and throughput falls off a cliff.

---

## How this differs from upstream

Upstream splits the project in two: a GPU container, and interactive driver
scripts that run on the host and reach into it with
`docker exec textymcspeechy-piper ...`.

On Unraid that split is the whole problem. The host is Slackware with a root
filesystem rebuilt in RAM at every boot, and it ships none of what the driver
scripts need — `tmux`, `bc`, `jq`, `inotify-tools`, `ffmpeg`. Anything you
install by hand is gone at the next reboot.

So this image puts both halves in one container. The interesting part is how the
dojo scripts survive that unchanged:

```
   upstream                          this image
   ────────                          ──────────
   host: run_training.sh             container: run_training.sh
     │  docker exec ─────┐             │  docker exec ──┐
     │                   ▼             │                ▼
     │        ┌──────────────────┐     │        /usr/local/bin/docker
     │        │ textymcspeechy-  │     │        (shim: strips `exec
     │        │ piper container  │     │         <container>`, runs
     │        └──────────────────┘     │         the rest locally)
     ▼                                 ▼
   two containers                    one container
```

A shim at `/usr/local/bin/docker` stands in for the Docker CLI and turns
`docker exec textymcspeechy-piper CMD` into plain local execution of `CMD`. That
means **upstream's scripts are byte-identical** — no patches to conflict with
every merge from upstream. The shim also swallows `docker stop
textymcspeechy-piper`, which the dojo issues when you quit the training console
and which here would kill the very container you're working in.

The shim is covered by [tests](tests/test-docker-shim.sh) that exercise every
call shape the dojo scripts actually use, run in CI before the image is built.

Other changes:

| | Upstream | This image |
|---|---|---|
| **Install** | Clone, `setup.sh`, build locally (20–45 min) | Pull from GHCR, apply template |
| **Host packages** | tmux, bc, jq, inotify-tools, ffmpeg | None |
| **Container user** | UID 1000, baked in at build time | `PUID`/`PGID`, default Unraid's 99:100 |
| **Dojo data** | Lives in the git checkout | Seeded into appdata on first start; survives updates |
| **Entry point** | Run scripts from the right directory | `tms` |

Upstream's `docker-compose.yml` and original `Dockerfile` are untouched and still
work — the entrypoint detects that compose runs it non-root and steps out of the
way.

## Updating

Pull a newer image from the Docker tab. Your appdata is never touched on
update — the dojo tree is seeded only when it's empty, so datasets, checkpoints,
voices and your edited `SETTINGS.txt` all survive.

The one thing that doesn't survive is **custom espeak-ng pronunciation rules**,
which get compiled into the container's own filesystem. Set
`AUTO_APPLY_CUSTOM_ESPEAK_RULES=true` in
`tts_dojo/ESPEAK_RULES/automated_espeak_rules.sh` and they are reapplied at the
start of every training run.

---

## Tuning

Per-voice settings live in `<voice>_dojo/scripts/SETTINGS.txt`, editable after
`tms new` creates the workspace.

| Setting | Default | Worth changing |
|---|---|---|
| `PIPER_BATCH_SIZE` | `5` | The biggest speed lever. A 16 GB card has room for 12–16; raise it in steps watching `nvidia-smi`. |
| `AUTO_SAVE_EVERY_NTH_EPOCH` | `2` | Each save is ~800 MB. On a small pool, 5–10. |
| `MINIMUM_DRIVE_SPACE_GB` | `20` | Raise it if the pool holds other appdata. |
| `FINE_TUNE_LR` | empty | Set (e.g. `0.0002`) when starting from a *finished* checkpoint whose learning rate has already decayed, or training barely moves. |

Training is hardcoded to FP32 (`precision=32` in `scripts/utils/piper_fit.py`),
which leaves a lot on the table on Blackwell. `precision="bf16-mixed"` is roughly
a 2× speedup and halves activation memory, but it's a real change to training
numerics that upstream hasn't validated — keep a known-good FP32 run first.

If you also run Plex or Frigate with `NVIDIA_VISIBLE_DEVICES=all`, a transcode
during training contends for the same VRAM. Pin both containers to explicit GPU
UUIDs.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `tms doctor` says no GPU visible | `--runtime=nvidia` missing from Extra Parameters, or the Nvidia-Driver plugin isn't installed. |
| `tms doctor` warns about the arch list | The PyTorch build has no kernels for your card. Open an issue — the image needs a newer torch. |
| `CUDA error: no kernel image is available` | Same as above, hit at runtime because `tms doctor` was skipped. |
| DataLoader workers die, bus errors | `--shm-size=8g` missing from Extra Parameters. |
| `DataLoader worker (pid N) is killed by signal` hours into a run | The `--memory` cap in Extra Parameters. The 8g of shm counts toward it, and a raised `PIPER_BATCH_SIZE` costs host RAM too. Raise the cap. |
| Console says the dojo isn't seeded | The appdata path isn't mapped, or isn't writable. |
| Voices are root-owned | `RUN_AS_ROOT=true` is set, or scripts were run directly as root instead of through `tms`. |
| Training slows to a crawl part way in | The mover started relocating the dojo. Set the share to Cache: Only. |
| Espeak pronunciation rules vanished after an update | Expected — see [Updating](#updating). |
| Pronunciation is wrong and you already trained | Espeak rules must be applied *before* training. Retraining is the only fix. |

## Building it yourself

```bash
git clone https://github.com/adman234/TextyMcSpeechy-unraid
cd TextyMcSpeechy-unraid
docker build -f Dockerfile.unraid -t textymcspeechy-unraid .
```

Then point the template's Repository field at `textymcspeechy-unraid` instead of
the GHCR tag. Expect 20–45 minutes; the build compiles PyTorch extensions and
`monotonic_align` from source.

---

MIT, same as upstream. Not affiliated with
[bacca87/TextyMcSpeechy](https://github.com/bacca87/TextyMcSpeechy),
[domesticatedviking/TextyMcSpeechy](https://github.com/domesticatedviking/TextyMcSpeechy),
Lime Technology, or NVIDIA.
