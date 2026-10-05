# Architecture

## Signal flow (per sample n)

```
 x1,x2 ──► beamformer + VAD ──► x_anc = x1 − speech_band ──► x_buf ──► w ──► y_anc ──┐
                 │                                                                   ▼
                 └─► isolated speech ───────────────────────────────────────► y_drive = y_anc + speech + ext
 e_m ──(− Ŝm*y)──► d̂m ──► Mv ──► d̂v ──(+ Ŝv(θ)*y)──► êv ──► FxLMS update of w
 yaw θ ──► Ŝv(θ) = (1−α)Ŝv(0°) + αŜv(45°),  α = clip(θ,0,45)/45
```

## Equations

- Disturbance at physical mic: `d̂m(n) = em(n) − Ŝm(z)·y(n)` (IMC: remove own anti-noise)
- Virtual disturbance: `d̂v(n) = Mv(z)·d̂m(n)`, with `Mv ≈ Pv/Pm`
- Virtual error: `êv(n) = d̂v(n) + Ŝv(z,θ)·y(n)`
- Filtered reference: `x'(n) = Ŝv(z,θ)·x(n)`
- Update: `w ← (1−ν)w − μ / (‖x'‖² + β) · êv(n) · x'`

## Calibration

`Mv` is designed offline by ridge least squares (`plants.ls_deconvolve`). `Ŝm`, `Ŝv(0°)`, `Ŝv(45°)` are loaded with `set_offline_paths`. In this repo they come from synthetic FIR plants (`plants.make_synthetic_plants`); replace with measured impulse responses for real use.

## Causality

Anti-noise must reach the eardrum no later than the noise. The synthetic primary path (noise → mic) is deliberately longer than the secondary path (driver → ear). With a short primary path the broadband component becomes non-causal and attenuation drops.
