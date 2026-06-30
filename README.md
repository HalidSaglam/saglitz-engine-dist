# Saglitz engine bundle (distribution)

Hosts the **self-contained engine bundle** that [Saglitz Photo Studio] downloads
on first launch — a portable CPython + ML stack + the GPL-3.0 command-line tools
(`draw-things-cli`, `ffmpeg`, `espeak-ng`). License texts ship inside each bundle.

Built reproducibly by `build-engine-bundle.sh` in the app repo. The app verifies
each download against a pinned SHA-256. Releases are tagged `engine-vN`.
