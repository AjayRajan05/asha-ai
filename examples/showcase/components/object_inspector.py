# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Object inspector component for ASHA Control Center."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set
import streamlit as st


def _safe_serialize(obj: Any, seen: Optional[Set[int]] = None) -> Any:
    """Recursively convert arbitrary Python dataclasses, models, or objects into JSON-serializable structures."""
    if seen is None:
        seen = set()

    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj

    obj_id = id(obj)
    if obj_id in seen:
        return f"<circular_reference: {type(obj).__name__}>"
    seen.add(obj_id)

    if isinstance(obj, Enum):
        return obj.value

    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v, seen) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_safe_serialize(item, seen) for item in obj]

    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return _safe_serialize(obj.model_dump(), seen)
        except Exception:
            pass

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _safe_serialize(obj.to_dict(), seen)
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return {k: _safe_serialize(v, seen) for k, v in vars(obj).items() if not k.startswith("_")}
        except Exception:
            pass

    return str(obj)


def safe_serialize_json(obj: Any, indent: int = 2) -> str:
    """Safely serialize any Python object to formatted JSON without crashing or recursing infinitely."""
    try:
        data = _safe_serialize(obj)
        return json.dumps(data, indent=indent, default=str)
    except Exception as ex:
        return json.dumps({"error": f"Serialization error: {ex}", "str_repr": str(obj)}, indent=indent)


def render_object_inspector(
    result_obj: Any = None,
    human_renderer: Optional[Callable[[], None]] = None,
    title: str = "Result Inspection",
    *,
    raw_obj: Any = None,
    human_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Render 3-tab inspector (Human View, Raw Object, JSON) for any ASHA result object."""
    target = raw_obj if raw_obj is not None else result_obj
    st.markdown(f"#### {title}")
    tab_human, tab_raw, tab_json = st.tabs(["Human View", "Raw Object", "JSON"])

    with tab_human:
        if human_renderer:
            human_renderer()
        elif human_data:
            st.dataframe([human_data], use_container_width=True, hide_index=True)
        else:
            if hasattr(target, "output"):
                st.markdown("**Output Content:**")
                st.code(str(target.output), language="markdown")
            elif hasattr(target, "sanitized"):
                st.markdown("**Sanitized Text:**")
                st.code(str(target.sanitized), language="markdown")
            elif hasattr(target, "optimized"):
                st.markdown("**Optimized Text:**")
                st.code(str(target.optimized), language="markdown")
            else:
                st.json(_safe_serialize(target))

    with tab_raw:
        st.code(repr(target), language="python")

    with tab_json:
        st.code(safe_serialize_json(target), language="json")
