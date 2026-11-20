# Wafer Daily

[Open the daily atlas](https://jasonjiang8866.github.io/waferVLM/). Every Singapore calendar day, each dataset contributes one image from its attributed preview pool. Everyone sees the same edition. The next image appears at midnight or when a sleeping tab returns. The date indexes each dataset independently, so all its previews appear before one repeats. Opening a card shows its source, a reading question and a full-size image link. Progress lives in the visitor's browser.

The microscope at `explorer.html` offers 12 Carinthia-S specimens with paired masks and the original silicon SEM cross-section. It supports class filters and adjustable mask overlays.

## Build

From the repo root:

```sh
python3 viewer/build.py dist/viewer
python3 -m http.server 8000 --directory dist/viewer
```

Open http://localhost:8000. To export the microscope as a self-contained HTML file, run `python3 viewer/build.py dist/offline --standalone`.

The daily builder checks the dataset IDs against `data_sources/cards/`. To add a collection, add its display metadata in `collections.json` and its attributed image pool in `daily-data/`. Local assets go in `daily-assets/`. The Carinthia-S pool reuses the checksum-verified original preview manifest. Some collections embed the authors’ own image URLs and need network access. Displaying a preview does not change the bulk dataset’s storage or license status.

GitHub Actions builds and checks the gallery and microscope on pull requests. Changes to viewer sources, sample files, data cards or the workflow on main trigger publication to GitHub Pages. Day changes run in the browser and need no scheduled deployment or GPU.
