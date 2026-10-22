"""
Structured annotation form component.

Renders a form for editing the fields of one ObjectAnnotation:
  category, material, condition, count, location, caption, sorting_action

Also renders the scene_context field at the image level when obj is None.

Returns the mutated ObjectAnnotation (or updated scene_context string),
or None if the form was cancelled.

Usage
-----
    updated = structured_form(obj, key_prefix="obj_o1")
    if updated is not None:
        rec.objects[i] = updated
        update_record(rec, db_path)
"""

from __future__ import annotations

import streamlit as st

from wastecap.schema import (
    Condition,
    Material,
    ObjectAnnotation,
    SortingAction,
    WasteCategory,
)
from wastecap.taxonomy import default_sorting_action

# Build sorted option lists once at module load (no I/O)
_CATEGORIES = [c.value for c in WasteCategory]
_MATERIALS = [m.value for m in Material]
_CONDITIONS = [c.value for c in Condition]
_ACTIONS = [a.value for a in SortingAction]


def structured_form(
    obj: ObjectAnnotation | None = None,
    *,
    key_prefix: str = "",
    scene_context: str | None = None,
) -> tuple[ObjectAnnotation | None, str | None]:
    """Render the structured annotation form.

    Args:
        obj:           Existing ObjectAnnotation to pre-populate, or None for
                       a blank form.
        key_prefix:    Unique key prefix for Streamlit widgets.
        scene_context: Current scene-context string (image-level field).

    Returns:
        (updated_obj, updated_scene_context)
        Both values are None if the user did not submit.
        updated_obj is None if there are no objects to edit.
    """
    with st.form(key=f"{key_prefix}_structured_form", clear_on_submit=False):
        st.subheader("Structured Fields")

        # ── Scene context (image-level) ──
        new_scene = st.text_area(
            "Scene context",
            value=scene_context or "",
            height=68,
            key=f"{key_prefix}_scene_ctx",
            help="One sentence: clutter level, occlusion, surroundings.",
            placeholder="e.g. outdoor, cluttered, multiple overlapping objects",
        )

        updated_obj: ObjectAnnotation | None = None

        if obj is not None:
            st.divider()
            st.caption(f"Object: `{obj.object_id}`")

            col1, col2 = st.columns(2)
            with col1:
                cat_idx = _CATEGORIES.index(obj.category.value)
                new_cat = st.selectbox(
                    "Category",
                    _CATEGORIES,
                    index=cat_idx,
                    key=f"{key_prefix}_cat",
                )
                cond_idx = (
                    _CONDITIONS.index(obj.condition.value) if obj.condition is not None else 0
                )
                new_cond = st.selectbox(
                    "Condition",
                    _CONDITIONS,
                    index=cond_idx,
                    key=f"{key_prefix}_cond",
                )
                new_count = st.number_input(
                    "Count",
                    min_value=1,
                    max_value=99,
                    value=max(1, obj.count),
                    step=1,
                    key=f"{key_prefix}_count",
                )
            with col2:
                mat_idx = _MATERIALS.index(obj.material.value)
                new_mat = st.selectbox(
                    "Material",
                    _MATERIALS,
                    index=mat_idx,
                    key=f"{key_prefix}_mat",
                )
                # Auto-suggest sorting action from (category, material) defaults
                suggested_action = default_sorting_action(new_cat, new_mat).value
                act_idx = (
                    _ACTIONS.index(obj.sorting_action.value)
                    if obj.sorting_action is not None
                    else _ACTIONS.index(suggested_action)
                )
                new_action = st.selectbox(
                    "Sorting action",
                    _ACTIONS,
                    index=act_idx,
                    key=f"{key_prefix}_action",
                    help="Auto-suggested from category + material; override if needed.",
                )

            new_location = st.text_input(
                "Location",
                value=obj.location or "",
                key=f"{key_prefix}_loc",
                placeholder="e.g. lower left, next to a can",
            )
            new_caption = st.text_area(
                "Object caption",
                value=obj.caption or "",
                height=80,
                key=f"{key_prefix}_cap",
                placeholder="<condition> <material> <object>, e.g. crushed plastic bottle",
            )

        submitted = st.form_submit_button("💾 Save", use_container_width=True)

        if not submitted:
            return None, None

        # Build updated ObjectAnnotation
        if obj is not None:
            updated_obj = obj.model_copy(
                update={
                    "category": WasteCategory(new_cat),
                    "material": Material(new_mat),
                    "condition": Condition(new_cond),
                    "count": int(new_count),
                    "location": new_location.strip() or None,
                    "sorting_action": SortingAction(new_action),
                    "caption": new_caption.strip() or None,
                }
            )

        return updated_obj, new_scene.strip() or None


def scene_context_widget(
    current: str | None,
    *,
    key: str = "scene_ctx_inline",
) -> str | None:
    """Inline (non-form) scene context text area.

    Returns updated value immediately on change (no submit button).
    Useful for the Curate page's quick-edit flow.
    """
    return (
        st.text_area(
            "Scene context",
            value=current or "",
            height=68,
            key=key,
            help="One sentence: clutter, occlusion, surroundings.",
            placeholder="e.g. outdoor, cluttered, multiple overlapping objects",
        )
        or None
    )
