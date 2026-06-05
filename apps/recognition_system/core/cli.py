import argparse
from pathlib import Path
from typing import Dict, List

import cv2

from .feature_db import FeatureDB
from .operations import build_runtime, draw_recognitions, iter_images, load_gallery, recognize_faces, register_dataset, register_image
from .runtime_io import append_jsonl, now_iso, write_csv


def emit_event(log_jsonl: str, event_type: str, payload: Dict) -> None:
    if not log_jsonl:
        return
    event = {
        "ts": now_iso(),
        "event_type": event_type,
        **payload,
    }
    append_jsonl(log_jsonl, event)


def add_runtime_args(parser: argparse.ArgumentParser, include_camera: bool = False, include_image: bool = False) -> None:
    parser.add_argument("--weights", required=True, help="Path to trained model_best.pt")
    parser.add_argument("--model-name", default="iresnet50")
    parser.add_argument("--db-path", default="face_system/face_features.db")
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--threshold", type=float, default=0.45, help="Cosine similarity threshold")
    parser.add_argument("--gallery-mode", default="mean", choices=["mean", "all"])
    parser.add_argument("--match-reduce", default="topk_mean", choices=["best", "mean", "topk_mean"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--det-conf-threshold", type=float, default=0.90, help="MTCNN confidence threshold")
    parser.add_argument("--det-min-size", type=int, default=40, help="Minimum detected face size in pixels")
    parser.add_argument("--detector-backend", default="mtcnn", choices=["mtcnn"], help="Face detector backend (currently only MTCNN is supported)")
    parser.add_argument("--yolo-weights", default="", help=argparse.SUPPRESS)  # Deprecated: not used (MTCNN backend only)
    if include_camera:
        parser.add_argument("--camera-id", type=int, default=0)
    if include_image:
        parser.add_argument("--image-path", required=True)
        parser.add_argument("--save-path", default="")


def cmd_register_dir(args) -> None:
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    model, detector = build_runtime(
        args.weights,
        args.model_name,
        args.img_size,
        args.device,
        args.det_conf_threshold,
        args.det_min_size,
        args.detector_backend,
        args.yolo_weights,
    )
    with FeatureDB(args.db_path) as db:
        results = register_dataset(
            db=db,
            model=model,
            detector=detector,
            dataset_dir=dataset_dir,
            clear_first=args.clear_first,
            max_images_per_person=args.max_images_per_person,
        )
        total_saved = 0
        for item in results:
            total_saved += item["saved"]
            print(
                f"[register-dir] {item['person_name']}: processed={item['processed']} saved={item['saved']} failed={item['failed']}"
            )
            emit_event(
                args.log_jsonl,
                "register_dir_person",
                {
                    "db_path": args.db_path,
                    "person_name": item["person_name"],
                    "processed": item["processed"],
                    "saved": item["saved"],
                    "failed": item["failed"],
                },
            )
        print(f"[done] persons={len(results)} total_embeddings={total_saved} db={args.db_path}")


def cmd_register_image(args) -> None:
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image_path not found: {image_path}")

    model, detector = build_runtime(
        args.weights,
        args.model_name,
        args.img_size,
        args.device,
        args.det_conf_threshold,
        args.det_min_size,
        args.detector_backend,
        args.yolo_weights,
    )
    with FeatureDB(args.db_path) as db:
        if args.clear_first:
            db.clear_person_embeddings(args.person_name)
        ok = register_image(db, model, detector, args.person_name, image_path)
        if not ok:
            raise RuntimeError(f"No face detected in image: {image_path}")
        print(f"[register-image] saved 1 embedding for {args.person_name} into {args.db_path}")
        emit_event(
            args.log_jsonl,
            "register_image",
            {
                "db_path": args.db_path,
                "person_name": args.person_name,
                "image_path": str(image_path),
                "saved": 1,
            },
        )


def cmd_recognize_image(args) -> None:
    frame = cv2.imread(args.image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {args.image_path}")

    model, detector = build_runtime(
        args.weights,
        args.model_name,
        args.img_size,
        args.device,
        args.det_conf_threshold,
        args.det_min_size,
        args.detector_backend,
        args.yolo_weights,
    )
    gallery = load_gallery(args.db_path, args.gallery_mode)
    if len(gallery) == 0:
        raise RuntimeError(f"Feature DB is empty: {args.db_path}")

    results = recognize_faces(frame, model, detector, gallery, args.threshold, args.match_reduce, args.topk)
    if len(results) == 0:
        print("No face found.")
        return

    annotated = draw_recognitions(frame, results)
    for item in results:
        print(item["label"])
        emit_event(
            args.log_jsonl,
            "recognize_image_face",
            {
                "db_path": args.db_path,
                "image_path": args.image_path,
                "name": item["name"],
                "score": round(float(item["score"]), 6),
                "accepted": bool(item["accepted"]),
                "box": [int(v) for v in item["box"]],
            },
        )

    if args.save_path:
        cv2.imwrite(args.save_path, annotated)
        print(f"Result saved to: {args.save_path}")
        return

    cv2.imshow("face_system_cli_image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cmd_recognize_camera(args) -> None:
    model, detector = build_runtime(
        args.weights,
        args.model_name,
        args.img_size,
        args.device,
        args.det_conf_threshold,
        args.det_min_size,
        args.detector_backend,
        args.yolo_weights,
    )
    gallery = load_gallery(args.db_path, args.gallery_mode)
    if len(gallery) == 0:
        raise RuntimeError(f"Feature DB is empty: {args.db_path}")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.camera_id}")

    print("Press 'q' to quit.")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame_idx += 1
        results = recognize_faces(frame, model, detector, gallery, args.threshold, args.match_reduce, args.topk)
        annotated = draw_recognitions(frame, results)
        for item in results:
            emit_event(
                args.log_jsonl,
                "recognize_camera_face",
                {
                    "db_path": args.db_path,
                    "frame_idx": frame_idx,
                    "name": item["name"],
                    "score": round(float(item["score"]), 6),
                    "accepted": bool(item["accepted"]),
                    "box": [int(v) for v in item["box"]],
                },
            )
        cv2.imshow("face_system_cli_camera", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def cmd_list_persons(args) -> None:
    with FeatureDB(args.db_path) as db:
        rows = db.list_persons()
        stats = db.get_stats()
    print(f"persons={stats['person_count']} embeddings={stats['embedding_count']}")
    for name, count in rows:
        print(f"{name}\t{count}")


def cmd_remove_person(args) -> None:
    with FeatureDB(args.db_path) as db:
        removed = db.delete_person(args.person_name)
    if not removed:
        raise RuntimeError(f"Person not found: {args.person_name}")
    print(f"[remove-person] removed {args.person_name} from {args.db_path}")


def cmd_recognize_dir(args) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    model, detector = build_runtime(
        args.weights,
        args.model_name,
        args.img_size,
        args.device,
        args.det_conf_threshold,
        args.det_min_size,
        args.detector_backend,
        args.yolo_weights,
    )
    gallery = load_gallery(args.db_path, args.gallery_mode)
    if len(gallery) == 0:
        raise RuntimeError(f"Feature DB is empty: {args.db_path}")

    rows: List[Dict] = []
    total_images = 0
    total_faces = 0

    for image_path in iter_images(input_dir):
        total_images += 1
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        results = recognize_faces(frame, model, detector, gallery, args.threshold, args.match_reduce, args.topk)
        total_faces += len(results)
        rel_path = str(image_path.relative_to(input_dir))

        for idx, item in enumerate(results):
            rows.append(
                {
                    "image_path": rel_path,
                    "face_index": idx,
                    "name": item["name"],
                    "score": round(float(item["score"]), 6),
                    "accepted": int(bool(item["accepted"])),
                    "box_x": int(item["box"][0]),
                    "box_y": int(item["box"][1]),
                    "box_w": int(item["box"][2]),
                    "box_h": int(item["box"][3]),
                }
            )

        if args.save_annotated_dir:
            out_path = Path(args.save_annotated_dir) / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            annotated = draw_recognitions(frame, results)
            cv2.imwrite(str(out_path), annotated)

        emit_event(
            args.log_jsonl,
            "recognize_dir_image",
            {
                "db_path": args.db_path,
                "image_path": rel_path,
                "face_count": len(results),
            },
        )

    if args.report_csv:
        write_csv(
            args.report_csv,
            rows,
            fieldnames=[
                "image_path",
                "face_index",
                "name",
                "score",
                "accepted",
                "box_x",
                "box_y",
                "box_w",
                "box_h",
            ],
        )
        print(f"report saved to: {args.report_csv}")

    print(f"[recognize-dir] images={total_images} faces={total_faces} records={len(rows)}")


def cmd_init_project(args) -> None:
    root = Path(args.root_dir)
    subdirs = [
        root / "db",
        root / "logs",
        root / "reports",
        root / "inputs" / "register_dataset",
        root / "inputs" / "test_images",
        root / "outputs" / "annotated",
    ]
    for path in subdirs:
        path.mkdir(parents=True, exist_ok=True)

    config_text = """{
  "weights": "/path/to/model_best.pt",
  "model_name": "iresnet50",
  "db_path": "face_system/face_features.db",
  "img_size": 112,
  "device": "auto",
  "threshold": 0.45,
  "gallery_mode": "mean",
  "match_reduce": "topk_mean",
  "topk": 3,
  "log_jsonl": "face_runtime/logs/events.jsonl"
}
"""
    config_path = root / "config.sample.json"
    if not config_path.exists():
        config_path.write_text(config_text, encoding="utf-8")

    print(f"[init-project] root={root}")
    print(f"[init-project] sample config={config_path}")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Complete face recognition system CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_dir_parser = subparsers.add_parser("register-dir", help="Register folder dataset: dataset_dir/person/*.jpg")
    register_dir_parser.add_argument("--dataset-dir", required=True)
    register_dir_parser.add_argument("--weights", required=True, help="Path to trained model_best.pt")
    register_dir_parser.add_argument("--model-name", default="iresnet50")
    register_dir_parser.add_argument("--db-path", default="face_system/face_features.db")
    register_dir_parser.add_argument("--img-size", type=int, default=112)
    register_dir_parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    register_dir_parser.add_argument("--det-conf-threshold", type=float, default=0.90, help="MTCNN confidence threshold")
    register_dir_parser.add_argument("--det-min-size", type=int, default=40, help="Minimum detected face size in pixels")
    register_dir_parser.add_argument("--detector-backend", default="mtcnn", choices=["mtcnn"], help="Face detector backend (currently only MTCNN is supported)")
    register_dir_parser.add_argument("--yolo-weights", default="", help=argparse.SUPPRESS)  # Deprecated: not used (MTCNN backend only)
    register_dir_parser.add_argument("--clear-first", action="store_true")
    register_dir_parser.add_argument("--max-images-per-person", type=int, default=0, help="0 means no limit")
    register_dir_parser.add_argument("--log-jsonl", default="", help="Optional audit log path")
    register_dir_parser.set_defaults(func=cmd_register_dir)

    register_image_parser = subparsers.add_parser("register-image", help="Register one image for one person")
    register_image_parser.add_argument("--person-name", required=True)
    register_image_parser.add_argument("--image-path", required=True)
    register_image_parser.add_argument("--weights", required=True, help="Path to trained model_best.pt")
    register_image_parser.add_argument("--model-name", default="iresnet50")
    register_image_parser.add_argument("--db-path", default="face_system/face_features.db")
    register_image_parser.add_argument("--img-size", type=int, default=112)
    register_image_parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    register_image_parser.add_argument("--det-conf-threshold", type=float, default=0.90, help="MTCNN confidence threshold")
    register_image_parser.add_argument("--det-min-size", type=int, default=40, help="Minimum detected face size in pixels")
    register_image_parser.add_argument("--detector-backend", default="mtcnn", choices=["mtcnn"], help="Face detector backend (currently only MTCNN is supported)")
    register_image_parser.add_argument("--yolo-weights", default="", help=argparse.SUPPRESS)  # Deprecated: not used (MTCNN backend only)
    register_image_parser.add_argument("--clear-first", action="store_true")
    register_image_parser.add_argument("--log-jsonl", default="", help="Optional audit log path")
    register_image_parser.set_defaults(func=cmd_register_image)

    recognize_image_parser = subparsers.add_parser("recognize-image", help="Recognize faces from one image")
    add_runtime_args(recognize_image_parser, include_image=True)
    recognize_image_parser.add_argument("--log-jsonl", default="", help="Optional audit log path")
    recognize_image_parser.set_defaults(func=cmd_recognize_image)

    recognize_camera_parser = subparsers.add_parser("recognize-camera", help="Recognize faces from webcam")
    add_runtime_args(recognize_camera_parser, include_camera=True)
    recognize_camera_parser.add_argument("--log-jsonl", default="", help="Optional audit log path")
    recognize_camera_parser.set_defaults(func=cmd_recognize_camera)

    recognize_dir_parser = subparsers.add_parser("recognize-dir", help="Offline recognize all images under a directory")
    recognize_dir_parser.add_argument("--input-dir", required=True)
    add_runtime_args(recognize_dir_parser)
    recognize_dir_parser.add_argument("--report-csv", default="", help="Optional CSV report path")
    recognize_dir_parser.add_argument("--save-annotated-dir", default="", help="Optional annotated images output dir")
    recognize_dir_parser.add_argument("--log-jsonl", default="", help="Optional audit log path")
    recognize_dir_parser.set_defaults(func=cmd_recognize_dir)

    list_persons_parser = subparsers.add_parser("list-persons", help="List all registered persons in DB")
    list_persons_parser.add_argument("--db-path", default="face_system/face_features.db")
    list_persons_parser.set_defaults(func=cmd_list_persons)

    remove_person_parser = subparsers.add_parser("remove-person", help="Remove one person and all embeddings from DB")
    remove_person_parser.add_argument("--db-path", default="face_system/face_features.db")
    remove_person_parser.add_argument("--person-name", required=True)
    remove_person_parser.set_defaults(func=cmd_remove_person)

    init_project_parser = subparsers.add_parser("init-project", help="Create a production-like runtime folder layout")
    init_project_parser.add_argument("--root-dir", default="face_runtime")
    init_project_parser.set_defaults(func=cmd_init_project)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()