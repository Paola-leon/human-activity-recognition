# Guía de uso de uv

Este repositorio usa [uv](https://docs.astral.sh/uv/) para gestionar Python y las dependencias. Con uv todos trabajamos con **las mismas versiones exactas** de cada librería, fijadas en `uv.lock`.

## Archivos del entorno

| Archivo | Qué es | ¿Se edita a mano? |
|---|---|---|
| `pyproject.toml` | Dependencias declaradas del proyecto | Sí (o con `uv add` / `uv remove`) |
| `uv.lock` | Versiones exactas resueltas para todas las plataformas | **No**, lo genera uv |
| `.python-version` | Versión de Python del proyecto (3.13) | Rara vez |
| `.venv/` | Entorno virtual local | No; **no se sube a git** |

`pyproject.toml`, `uv.lock` y `.python-version` **sí se suben a git**.

---

## 1. Instalar uv (una sola vez)

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verifica con `uv --version`. No necesitas instalar Python por separado: si no tienes 3.13, uv lo descarga automáticamente (o puedes forzarlo con `uv python install 3.13`).

## 2. Crear el entorno (primera vez tras clonar)

PyTorch se instala **eligiendo uno de dos extras** según tu equipo:

| Tu equipo | Comando |
|---|---|
| Mac (Apple Silicon o Intel) o PC **sin** GPU NVIDIA | `uv sync --extra cpu` |
| Linux / Windows **con** GPU NVIDIA | `uv sync --extra cu128` |

```bash
git clone <url-del-repo>
cd human-activity-recognition
uv sync --extra cpu        # o --extra cu128
```

Esto crea `.venv/` con Python 3.13 y todas las dependencias. El entorno se llama **`human-activity-recognition`** (uv toma el nombre del proyecto) y así lo verás en VS Code y en la terminal.

> **GPU NVIDIA:** el extra `cu128` usa CUDA 12.8 y requiere driver NVIDIA **≥ 570**. Revisa tu versión con `nvidia-smi`. No hace falta instalar el CUDA Toolkit; las librerías CUDA vienen dentro de las wheels de PyTorch.
>
> **Mac:** el extra `cpu` en Mac instala el torch estándar, que incluye aceleración **MPS** en Apple Silicon (`torch.device("mps")`).

Verifica la instalación:
```bash
uv run python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available(), 'MPS:', torch.backends.mps.is_available())"
```

> ⚠️ **Siempre incluye tu extra al hacer `uv sync`.** Un `uv sync` sin `--extra` elimina torch del entorno (sync deja el entorno *exactamente* igual a lo pedido). Si te pasa, vuelve a correr `uv sync --extra cpu` (o `cu128`).

## 3. Uso diario

### Ejecutar código
```bash
uv run python script.py
uv run jupyter lab
```
`uv run` usa el entorno del proyecto automáticamente; no necesitas activarlo.

Si prefieres activarlo:
```bash
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
deactivate                     # para salir
```
Al activarlo, la terminal muestra `(human-activity-recognition)` al inicio de la línea.

### Notebooks en VS Code
1. Abre un `.ipynb` y haz clic en **Select Kernel** (arriba a la derecha).
2. Elige **Python Environments…** → **`human-activity-recognition (3.13.x)`** (la ruta debe apuntar a `.venv` dentro de este repositorio). El número exacto después de 3.13 depende de tu instalación.

Los notebooks se crearon con el kernel `base` de conda; cambia al kernel `human-activity-recognition` para usar las versiones del proyecto.

### Notebooks en JupyterLab
```bash
uv run jupyter lab
```
El kernel `Python 3` que aparece ya es el del `.venv`.

## 4. Después de hacer `git pull`

Si alguien cambió dependencias (`pyproject.toml` o `uv.lock` aparecen en el pull):
```bash
uv sync --extra cpu        # o --extra cu128
```

## 5. Agregar o quitar dependencias

```bash
uv add seaborn                 # agrega la última versión compatible
uv add "tqdm>=4.66"            # con restricción de versión
uv remove seaborn
```
Esto actualiza `pyproject.toml` y `uv.lock` a la vez. Después:

1. Corre de nuevo `uv sync --extra cpu` (o `cu128`) para asegurar que torch sigue instalado.
2. Haz commit de **ambos** archivos:
   ```bash
   git add pyproject.toml uv.lock
   git commit -m "chore: agregar seaborn"
   ```

**Reglas del equipo**
- No uses `pip install` dentro del `.venv`: esa instalación no queda registrada y los demás no la tendrán.
- No edites `uv.lock` a mano. Si hay conflicto de merge en `uv.lock`, resuelve primero `pyproject.toml` y luego ejecuta `uv lock` para regenerarlo.
- **No agregues torch con `uv add torch`**: ya está configurado en los extras `cpu` / `cu128`.

## 6. Actualizar versiones

```bash
uv lock --upgrade-package pandas   # solo un paquete
uv lock --upgrade                  # todos (con cuidado)
uv sync --extra cpu
```
Luego haz commit de `uv.lock`.

## 7. Referencia rápida

| Quiero… | Comando |
|---|---|
| Crear / actualizar el entorno | `uv sync --extra cpu` o `uv sync --extra cu128` |
| Correr un script | `uv run python archivo.py` |
| Abrir Jupyter | `uv run jupyter lab` |
| Agregar librería | `uv add <paquete>` |
| Quitar librería | `uv remove <paquete>` |
| Ver dependencias | `uv tree` |
| Regenerar el lock | `uv lock` |
| Borrar y recrear el entorno | `rm -rf .venv && uv sync --extra cpu` |
| Exportar a requirements.txt | `uv export --extra cpu --no-hashes > requirements.txt` |

## 8. Problemas comunes

**`ModuleNotFoundError: No module named 'torch'`**
Hiciste `uv sync` sin `--extra`. Corre `uv sync --extra cpu` (o `cu128`).

**`torch.cuda.is_available()` devuelve `False` en un equipo con GPU NVIDIA**
- Confirma que instalaste con `--extra cu128`: `uv run python -c "import torch; print(torch.__version__)"` debe terminar en `+cu128`.
- Revisa que el driver sea ≥ 570 con `nvidia-smi`.

**El notebook no encuentra librerías**
Tienes seleccionado otro kernel (por ejemplo `base` de conda). Cambia al kernel `human-activity-recognition (3.13.x)` del repo.

**`Extras 'cpu' and 'cu128' are incompatible`**
Solo se puede usar uno de los dos extras a la vez; elige el que corresponde a tu equipo.

**¿Por qué torch está fijado en 2.11?**
Es la versión más reciente publicada para CUDA 12.8. La fijamos en `pyproject.toml` (`torch==2.11.*`) para que Mac, CPU y GPU usen la misma versión. Si se cambia a una versión mayor de CUDA, se puede subir junto con el extra.
