# human-activity-recognition

## Entorno de desarrollo

El proyecto usa [uv](https://docs.astral.sh/uv/) (Python 3.13). Para configurarlo:

```bash
uv sync --extra cpu     # Mac o equipos sin GPU NVIDIA
uv sync --extra cu128   # Linux/Windows con GPU NVIDIA
```

Consulta la guía completa en [documentacion/GUIA_UV.md](documentacion/GUIA_UV.md).

## Datos

Usamos el dataset **Manual Material Handling Dataset for Biomechanical and Ergonomics Analysis** (Bassani, Filippeschi y Avizzano, Scuola Superiore Sant'Anna, 2021), publicado en Zenodo con licencia CC BY 4.0:

- Descarga: https://zenodo.org/records/4633087
- DOI: [10.5281/zenodo.4633087](https://doi.org/10.5281/zenodo.4633087)

Los datos **no se versionan en git** (`datos/` y `datos_unzipped/` están en `.gitignore`). Cada colaborador debe descargarlos y colocarlos en la raíz del repo con esta estructura exacta, porque los notebooks buscan esas rutas:

```
human-activity-recognition/
├── datos/
│   ├── Subjs_info.xlsx            ← suelto aquí, NO dentro de datos_unzipped/
│   └── *.7z                       ← archivos comprimidos originales (opcional conservarlos)
└── datos_unzipped/
    ├── Data/Subj_01 … Subj_14/         ← sEMG: *_EMG_data.csv (679 archivos, ~10 GB)
    ├── Labels/Subj_01 … Subj_14/       ← etiquetas en frames de sEMG: *_EMG_labels.csv, *_EMG_range.csv (679)
    ├── Labels 2/Subj_01 … Subj_14/     ← etiquetas en frames de MVNX: *_mvnx_labels.csv (517)
    ├── mvnx_files/Subj_01 … Subj_14/   ← captura de movimiento Xsens: *.mvnx (517 archivos, ~53 GB)
    └── m_files/                        ← scripts MATLAB de los autores (EMG_processing.m, Process.m)
```

Archivos de Zenodo y dónde va cada uno:

| Archivo en Zenodo | Tamaño | Se descomprime en | Notas |
|---|---|---|---|
| `EMG_Data.7z` | 4.1 GB | `datos_unzipped/Data/` | |
| `EMG_Labels.7z` | 54 kB | `datos_unzipped/Labels/` | |
| `mvnx_Labels.7z` | 37 kB | `datos_unzipped/Labels 2/` | El nombre `Labels 2` (con espacio) es el que usan los notebooks |
| `mvnx_files.7z` | 13.7 GB | `datos_unzipped/mvnx_files/` | Ocupa ~53 GB descomprimido |
| `m_files.7z` | 2 kB | `datos_unzipped/m_files/` | |
| `Subjs_info.7z` | 7 kB | `datos/Subjs_info.xlsx` | Mover el `.xlsx` a `datos/` |
| `mvn_files.7z` | 42.4 GB | — | **No se necesita**: son las mismas grabaciones en el formato binario nativo de Xsens (`.mvn`) |

Espacio necesario: ~18 GB de descarga y ~63 GB descomprimido. Los `.7z` se abren con [7-Zip](https://www.7-zip.org/) (Windows), `brew install sevenzip` (macOS) o `sudo apt install p7zip-full` (Linux).

Al descomprimir, revisa que **no quede una carpeta intermedia** (por ejemplo `datos_unzipped/EMG_Data/Data/…`). Si pasa, mueve el contenido un nivel arriba. Los nombres distinguen mayúsculas y minúsculas.

Para verificar la estructura, desde la raíz del repo:

```bash
ls datos/Subjs_info.xlsx
ls datos_unzipped    # debe mostrar: Data  Labels  Labels 2  m_files  mvnx_files
```

El notebook [`notebooks/02_Contenido_datos_unzipped.ipynb`](notebooks/02_Contenido_datos_unzipped.ipynb) recorre estas carpetas y explica qué contiene cada una.
