# WasteCap Annotation Guidelines

## Caption style

Describe only visible evidence.

- Object caption: `<condition> <material> <object>`, with a short location when
  useful.
- Scene caption: one sentence about clutter, occlusion, and surroundings.
- Use `unknown` instead of guessing material or condition.
- Keep captions concrete and grammatical; do not add disposal advice to the
  image-level caption.
- Counts describe visible instances, not assumed contents.

### Examples

| Good | Why |
|---|---|
| `crushed plastic bottle, lower left` | Visible condition, material, object, and location. |
| `intact metal can beside the cardboard box` | Uses a visible relation without speculation. |
| `A cluttered outdoor scene with a partially occluded wrapper.` | Concise scene context. |
| `unknown material bottle on the pavement` | Makes uncertainty explicit. |

| Counter-example | Problem | Correction |
|---|---|---|
| `recyclable plastic bottle` | “Recyclable” is not necessarily visible. | `plastic bottle` |
| `glass bottle` when material is unclear | Guesses material. | `unknown material bottle` |
| `three cans` when one can is visible | Invents a count. | `one can` or `a can` |
| `clean bottle` when dirt cannot be judged | Adds an unsupported attribute. | Omit `clean`. |

## Taxonomy

Categories are `bottle`, `can`, `cup`, `carton`, `bag`, `wrapper`,
`container`, `box`, `straw`, `lid`, `tube`, `food_waste`, `electronic`,
`clothing`, `other`, and `unknown`.

Materials are `plastic`, `metal`, `paper`, `cardboard`, `glass`, `organic`,
`textile`, `mixed`, and `unknown`.

Conditions are `intact`, `crushed`, `torn`, `dirty`, `deformed`, and
`partially_occluded`. Use only conditions supported by the image. A partially
occluded object may also be dirty or deformed when both are visible.

Sorting actions are `recycle_plastic`, `recycle_metal`, `recycle_paper`,
`recycle_cardboard`, `recycle_glass`, `compost`, `landfill`, and `hazardous`.
If the material or action cannot be supported, leave it unresolved rather than
guessing.

## Grounding and instructions

Every object instruction must refer to the same `object_id` as its bbox or
mask. Use a spatial phrase tied to the image:

> Pick up the crushed plastic bottle at the lower left and place it in the
> plastic recycling bin.

Do not approve an instruction with a missing, negative, zero-area, or
out-of-bounds bbox. Qwen-predicted boxes are evidence for evaluation only;
they do not replace the human reference box.

## Hallucination rules

Set `review.hallucination = true` when the candidate names an object,
material, count, condition, location, or other attribute that is not visible.
Examples:

- The image shows a can but the caption says “bottle”.
- The object is visually ambiguous but the caption asserts “glass”.
- One visible wrapper is described as “four wrappers”.
- The caption claims a logo, brand, or liquid that cannot be seen.

Do not flag a caption merely because it uses a harmless synonym or omits a
visible object; omission is a completeness issue unless it introduces a false
claim. Record the reason in reviewer notes.

## Review checklist

Before approval, verify:

1. The image-level caption describes only visible content.
2. Object IDs are unique and object fields match the image.
3. Material and condition are explicit or `unknown`.
4. Count and location are supported.
5. Every instruction points to an existing valid bbox/mask.
6. The hallucination flag and reviewer notes are recorded.
