# Wafer image explorer

Browse 12 Carinthia-S SEM images with their paired masks and one silicon-wafer SEM cross-section. Selecting Overlay displays the mask luminance in orange with adjustable opacity. Class numbers follow the source annotation. Original and Mask show the unmodified source files. Each specimen links to its source record and license.

From the repository root, build with Python 3 and serve the output directory:

```sh
python3 viewer/build.py dist/viewer
python3 -m http.server 8000 --directory dist/viewer
```

Open http://localhost:8000. The builder verifies each file against the public preview manifest. For a portable single-file version, build to a fresh directory with `--standalone` and open its `index.html`.

**GitHub Pages.** After merge, select GitHub Actions in Settings → Pages and run the Wafer image explorer workflow on main. It publishes only the built viewer. The app uses relative asset URLs for project subpaths.

**Cloudflare Pages.** From the repository root use `python3 viewer/build.py dist/viewer` as the build command and `dist/viewer` as the output directory. Each host serves static assets without RunPod access.
