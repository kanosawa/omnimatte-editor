# omnimatte-editor

A demo app for removing a specified object together with its associated regions (shadows, reflections, etc.) from a video. Masks are produced with [SAM2](https://github.com/facebookresearch/sam2), and the object is removed with [Generative Omnimatte](https://github.com/kanosawa/gen-omnimatte-public). [Detectron2](https://github.com/facebookresearch/detectron2) is used for preprocessing.

<video src="https://github.com/user-attachments/assets/962599d9-9981-4f6f-9e19-401f9afd28a8" width="100%" controls autoplay loop muted></video>

## Usage

The backend runs on a GPU server and the frontend runs on a local machine, connecting through an SSH tunnel. Tested with Ubuntu for the backend and Windows for the frontend; other environments may also work.

- Backend: [backend/README.md](backend/README.md)
- Frontend: [frontend/README.md](frontend/README.md)

## Acknowledgments

Thanks to the authors of [Detectron2](https://github.com/facebookresearch/detectron2), [SAM2](https://github.com/facebookresearch/sam2), and [Generative Omnimatte](https://gen-omnimatte.github.io/).

## Citation

If you use this software, please cite it as:

```bibtex
@software{kanosawa_omnimatte_editor_2026,
  author = {kanosawa},
  title  = {omnimatte-editor},
  year   = {2026},
  url    = {https://github.com/kanosawa/omnimatte-editor}
}
```