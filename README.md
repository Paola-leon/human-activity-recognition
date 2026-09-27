# human-activity-recognition

## Entorno de desarrollo

El proyecto usa [uv](https://docs.astral.sh/uv/) (Python 3.13). Para configurarlo:

```bash
uv sync --extra cpu     # Mac o equipos sin GPU NVIDIA
uv sync --extra cu128   # Linux/Windows con GPU NVIDIA
```

Consulta la guía completa en [documentacion/GUIA_UV.md](documentacion/GUIA_UV.md).
