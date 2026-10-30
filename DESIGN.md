# Design

<!-- impeccable:design-schema 1 -->

## Conversion Dashboard

The conversion dashboard is an operate-mode local workbench for long-running,
resource-intensive model conversion.

- **Scene:** An operator is managing a local GPU workstation, often in a dim
  environment, so the app uses a charcoal field rather than a generic light
  admin surface.
- **Hierarchy:** A large condensed title anchors the tool. The left work area
  is reserved for one new conversion, while the right queue exposes live
  status and raw converter output without changing pages.
- **Color:** Off-white text on dark green-charcoal surfaces; electric lime
  identifies the active action and running work. Success, cancellation, and
  failure retain distinct high-contrast status colors.
- **Controls:** Square, bordered fields and flat status bands make paths,
  settings, and console output feel like equipment labels rather than
  decorative cards.
- **Responsive behavior:** The queue moves below the form below 900px, and
  form controls become one column below 620px.
