# Prompt Canvas Editor

A single-file browser editor for building Ideogram-style structured JSON prompts with canvas-based bounding boxes. The app lives entirely in `ui.html`; there is no build step, package manager, or local server requirement.

## Features

- Set canvas width and height with sliders or by double-clicking the displayed values.
- Draw, move, resize, delete, and edit bounding boxes directly on the canvas.
- Cycle through selected boxes with the `<-` and `->` controls when boxes overlap.
- Edit global prompt fields, including high-level description, aesthetics, lighting, medium, style/photo mode, background, and color palette.
- Edit per-box mode, description, optional text content, and per-box color palette.
- Generate formatted JSON from the current canvas and form state.
- Paste existing prompt JSON into the JSON box and load it back into the editable canvas.

## Usage

Open `ui.html` directly in a modern browser.

The Tailwind design system is loaded from the Tailwind CDN, so the page needs internet access for styling. The editor logic itself is plain HTML, CSS, and JavaScript.

## Basic Workflow

1. Set the canvas size.
2. Draw boxes on the canvas by clicking and dragging.
3. Select a box and edit its properties in the right panel.
4. Fill in the global prompt settings.
5. Click `Generate JSON` to write the prompt JSON into the textarea.
6. Copy or save the generated JSON wherever your workflow needs it.

To edit an existing prompt, paste the JSON into the textarea and click `Load JSON`. The editor will rebuild the canvas boxes and form fields from the prompt.

## JSON Shape

The editor expects prompt JSON in this general form:

```json
{
  "high_level_description": "",
  "style_description": {
    "aesthetics": "",
    "lighting": "",
    "medium": "",
    "art_style": "",
    "color_palette": []
  },
  "compositional_deconstruction": {
    "background": "",
    "elements": [
      {
        "type": "obj",
        "bbox": [0, 0, 1000, 1000],
        "desc": "",
        "color_palette": []
      }
    ]
  }
}
```

Bounding boxes use normalized coordinates from `0` to `1000` in `[y1, x1, y2, x2]` order. The editor converts those coordinates to the current canvas size when loading JSON, then converts them back to normalized coordinates when generating JSON.