# Póster del proyecto

Póster A1 horizontal basado en la plantilla institucional (`poster_plantilla.tex`).

| Archivo | Qué es |
|---|---|
| `poster_proyecto.tex` | Fuente del póster del equipo |
| `poster_proyecto.pdf` | Póster compilado |
| `fig_ventana.png`, `fig_pocas.png` | Figuras 1 y 4, generadas con `generar_figuras.py` |
| `logo.jpeg`, `logo_der.png` | Logos del Tec y de la Scuola Superiore Sant'Anna |
| `poster_plantilla.tex`, `poster.pdf` | Plantilla original y su PDF de ejemplo (sólo referencia) |

## Compilar

**Overleaf:** sube la carpeta `Poster/` completa, elige `poster_proyecto.tex` como documento principal y compila con pdfLaTeX. No requiere nada más.

**Local** (TeX Live o MacTeX completos):

```bash
cd Poster
latexmk -pdf poster_proyecto.tex
latexmk -c        # borra los archivos auxiliares
```

**Local con BasicTeX** (instalación mínima en Mac): le faltan algunos paquetes que usa la plantilla. Se instalan una sola vez, sin `sudo`, en la carpeta del usuario:

```bash
tlmgr init-usertree
tlmgr --usermode install charter helvetic tikzfill pdfcol listingsutf8
```

Si aparece otro `File '....sty' not found`, se busca el paquete con `tlmgr --usermode search --global --file /nombre.sty` y se instala igual.

## Regenerar las figuras

Sólo hace falta si se cambian. Desde la raíz del repositorio:

```bash
uv run python Poster/generar_figuras.py
```

`fig_pocas.png` usa las medias del experimento de pocas etiquetas, escritas en el script. `fig_ventana.png` lee la caché sincronizada a 12 Hz (`notebooks/_cache/sync/fs12`), que no se versiona: se genera con el notebook de sincronización a partir de los datos de Zenodo.
