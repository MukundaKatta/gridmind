# gridmind

**GridMind — Building Energy Optimizer. AI-powered HVAC and energy optimization for commercial buildings.**

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-proprietary-red)

## Install
```bash
pip install -e ".[dev]"
```

## Quick Start
```python
from src.core import Gridmind
 instance = Gridmind()
r = instance.generate(input="test")
```

## CLI
```bash
python -m src status
python -m src run --input "data"
```

## API
| Method | Description |
|--------|-------------|
| `generate()` | Generate |
| `create()` | Create |
| `validate()` | Validate |
| `preview()` | Preview |
| `export()` | Export |
| `get_templates()` | Get templates |
| `get_stats()` | Get stats |
| `reset()` | Reset |

## Test
```bash
pytest tests/ -v
```

## License
(c) 2026 Officethree Technologies. All Rights Reserved.
