# Veo Prompt Guide

## Model: veo-3.1-fast-generate-preview

**Parameters:**
- `durationSeconds`: 4–8 (inclusive)
- `aspectRatio`: `16:9` (landscape), `9:16` (portrait/short-video), `1:1` (square)
- Image input: PNG/JPEG/WebP, used as composition/style reference (not strict adherence)

**API limits:**
- Video size: typically 2–15MB depending on duration and content
- Image input: base64 inline, max ~10MB source image
- Single video per request (no batch)

## Prompt Structure (Best Practices)

Strong Veo prompts follow this pattern:

```
[Shot type]. [Subject + action]. [Environment]. [Motion/timing]. [Lighting]. [Style/quality].
```

**Example (ASMR style):**
```
Top-down macro cinematic shot. Female hands with pink nails squeeze a hyperrealistic red strawberry 
over white marble. The berry shatters into fine sand-like powder in slow motion. 
Studio lighting, 4K, satisfying texture detail, high frame rate.
```

## Effective Shot Types
- `extreme close-up macro` — textures, tiny subjects
- `top-down overhead shot` — flat lays, cooking, hands
- `eye-level medium shot` — faces, conversations  
- `low angle` — dramatic, subject feels large
- `tracking shot` — follows moving subject
- `static locked-off shot` — environmental, slow reveals

## Motion Descriptors
- `slow motion`, `ultra slow motion`, `ramped slow motion`
- `smooth camera push-in / pull-out`
- `hand-held`, `stabilized`, `floating dolly`
- `360 orbit`, `bird's eye descend`

## Lighting
- `studio lighting, key + fill + rim`
- `natural window light, soft diffused`
- `golden hour, warm backlight`
- `neon-lit, cyberpunk atmosphere`
- `dark moody, single candle light`

## Style Modifiers
- `cinematic, film grain, anamorphic lens flare`
- `documentary, natural color grading`
- `anime-style illustration`, `3D render`
- `hyperrealistic, photorealistic 4K`
- `ASMR aesthetic, satisfying textures`

## Image Reference Tips
When passing an image to Veo:
- The image is used as **composition/style reference**, not strict adherence
- Works best for: color palette, framing, lighting setup, object composition
- Less reliable for: face identity, exact text, fine object details
- Best image formats: clean, well-lit, high contrast subjects work better
