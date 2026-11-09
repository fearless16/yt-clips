# Approved stadium artwork — production asset

`overlay.png` is the normalized RGBA artwork from the user-accepted v7 layout.
It is an AI-assisted empty-template edit of the owner's supplied image, not a
generated face/gameplay frame. Exact PNG SHA-256:

`f6a8fcd86c2f4082f12d2341d06bdb253e0ab47b156801315d201227a8efcdd5`

Geometry and validation live in `automation/stadium_layout.py`. Production does
not import private renderer scripts or read artwork from `temp/` or `shorts/`.
Both apertures use identical corner masks and 992×558 exact 16:9 source planes.
Caption plaque is artwork; the caption renderer supplies only 96px white/cyan ink.

Arial Black is read from the licensed Windows installation at
`%WINDIR%/Fonts/ariblk.ttf` and hash checked against the reviewed font. It is NOT
redistributed here. No fallback font is silently substituted. Another font or
machine needs explicit design/font review before changing the locked fingerprint.

Production retains a publisher disclaimer in the bottom 72px from frame one,
below GAMEPLAY VISUAL. That compliance strip was absent from the private final
pair and requires separate finished-file visual/UI review. No source/recording
permission is conveyed by this artwork or by visual acceptance.
