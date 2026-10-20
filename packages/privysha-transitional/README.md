# privysha (Deprecated Package)

> **Notice:** The `privysha` package has been renamed to **[`asha-si`](https://pypi.org/project/asha-si/)**.

Please update your project dependencies:

```bash
# Old:
pip install privysha

# New:
pip install asha-si
```

And update your Python imports:

```python
# Old:
import privysha

# New:
from asha import process, sanitize, optimize, wrap_llm, anchor
```

Installing this package will automatically install `asha-si` as a dependency. All APIs from `privysha` remain available as backward-compatible shims forwarding directly to `asha`.

- **GitHub Repository:** [https://github.com/AjayRajan05/asha-si](https://github.com/AjayRajan05/asha-si)
- **Documentation:** [https://ajayrajan05.github.io/asha-si/](https://ajayrajan05.github.io/asha-si/)
