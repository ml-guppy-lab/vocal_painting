# Vocal Painter — Flowchart

```mermaid
flowchart TD
    A([▶ Start]) --> B

    subgraph SETUP ["⚙️ Setup"]
        B[Create black canvas\n800 × 1200 px]
        B --> C[Start audio thread\nsounddevice mic stream opens]
        C --> D[Open OpenCV window]
        D --> E[Init orbit state\nangle=0, cx/cy=center]
    end

    E --> LOOP

    subgraph LOOP ["🔁 Main Draw Loop  —  runs forever"]
        F[cv2.waitKey — read keyboard]
        F --> G{Q or ESC\npressed?}
        G -- Yes --> QUIT
        G -- No --> H{SPACE\npressed?}
        H -- Yes --> I[Clear canvas\nReset angle & bloom]
        I --> J
        H -- No --> J{Window\nclosed?}
        J -- Yes --> QUIT
        J -- No --> K{Any brush\nin queue?}
        K -- No --> L[imshow current painting\nkeep window alive]
        L --> F
        K -- Yes --> M[Pull brush dict\nfrom queue]
    end

    subgraph RADIUS ["📐 Compute Radius"]
        M --> N[color, thickness from brush]
        N --> O[pitch_var = interp y → ±130 px\nhigh pitch = outward push]
        O --> P[wobble = amp × 50 × sin\nangle × 6 bumps per orbit]
        P --> Q{thickness ≥ 25\nfor ≥ 1 second?}
        Q -- Yes --> R[🌸 Bloom burst!\nbloom_boost surges +90 px\ndecays over 2.5 s]
        Q -- No --> S
        R --> S[radius = 150 + pitch_var\n+ wobble + bloom_boost]
    end

    subgraph DRAW ["🖌️ Draw Stroke"]
        S --> T[polar → pixel\nx = cx + r·cos θ\ny = cy + r·sin θ]
        T --> U{prev_pt\nexists?}
        U -- Yes --> V[cv2.line\nprev_pt → curr_pt]
        U -- No --> W[cv2.circle\nsingle dot]
        V --> X[angle += 0.04 rad]
        W --> X
        X --> Y[imshow updated painting]
        Y --> F
    end

    subgraph QUIT ["🛑 Shutdown"]
        QUIT([Quit triggered]) --> Z1[Stop audio thread]
        Z1 --> Z2[Save painting\nartwork/painting_NNN.png]
        Z2 --> Z3[Destroy OpenCV window\nflush macOS event queue]
        Z3 --> Z4([✅ Done])
    end
```

## How each voice feature maps to the brush

| Voice feature | How it's measured | What it controls |
|---|---|---|
| **Pitch** (Hz) | YIN algorithm in `vocal.py` | Radius — high note = brush moves outward |
| **Amplitude** (RMS) | Root-mean-square of the audio frame | Stroke thickness + wobble depth + bloom arming |
| **Spectral centroid** (Hz) | Weighted average frequency | Stroke colour — bright/high = warm (red/orange), dull/low = cool (blue/violet) |
