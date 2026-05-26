import argparse
import numbers
import os

import mxnet as mx
import numpy as np
import torch
import torch.nn.functional as F

from backbones import get_model
from utils.utils_config import get_config


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def to_int_tuple(value, length):
    if isinstance(value, str):
        value = [int(item.strip()) for item in value.split(",")]
    value = tuple(int(item) for item in value)
    if len(value) != length:
        raise ValueError(f"Expected {length} values, got {value}")
    return value


def normalize_source_labels(source_labels, context):
    if source_labels is None:
        return None
    source_labels = list(source_labels)
    if len(source_labels) == 0:
        raise ValueError(
            f"{context} = [] is ambiguous. "
            "Use None to allow all non-target classes, or provide an explicit label list."
        )
    return set(int(label) for label in source_labels)


def same_path(left, right):
    if left is None or right is None:
        return False
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def resolve_model_path(model_path, output_dir):
    if model_path is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    search_roots = [output_dir]
    if not os.path.isabs(output_dir):
        search_roots.append(os.path.join(os.path.dirname(__file__), output_dir))

    for candidate_root in search_roots:
        primary = os.path.join(candidate_root, "model.pt")
        if os.path.exists(primary):
            return primary

    candidate_models = []
    for candidate_root in search_roots:
        candidate_root = os.path.normpath(candidate_root)
        parent_dir = os.path.dirname(candidate_root) or "."
        output_prefix = os.path.basename(candidate_root)
        if not os.path.isdir(parent_dir):
            continue
        for entry in os.listdir(parent_dir):
            candidate_dir = os.path.join(parent_dir, entry)
            candidate_model = os.path.join(candidate_dir, "model.pt")
            if os.path.isdir(candidate_dir) and entry.startswith(output_prefix) and os.path.exists(candidate_model):
                candidate_models.append(candidate_model)

    if candidate_models:
        candidate_models.sort(
            key=lambda path: (os.path.getmtime(path), os.path.normcase(path)),
            reverse=True,
        )
        return candidate_models[0]

    raise FileNotFoundError(
        f"Could not find model.pt under {output_dir!r}, and no dated output directory "
        "with the same prefix contained a model.pt."
    )


class RecordReader:
    def __init__(self, root_dir):
        path_imgrec = os.path.join(root_dir, "train.rec")
        path_imgidx = os.path.join(root_dir, "train.idx")
        if not os.path.exists(path_imgrec) or not os.path.exists(path_imgidx):
            raise FileNotFoundError(f"RecordIO files not found under {root_dir}")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        header, _ = mx.recordio.unpack(self.imgrec.read_idx(0))
        self.header0 = None
        self.identity_start = None
        self.identity_end = None
        self._label_index_cache = {}
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.identity_start = self.header0[0]
            self.identity_end = self.header0[1]
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def read(self, idx):
        header, img = mx.recordio.unpack(self.imgrec.read_idx(int(idx)))
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        image = mx.image.imdecode(img).asnumpy()
        return int(label), image

    def shuffled_indices(self, seed):
        rng = np.random.RandomState(seed)
        indices = self.imgidx.copy()
        rng.shuffle(indices)
        return indices

    def get_label_indices(self, label):
        label = int(label)
        if label in self._label_index_cache:
            return self._label_index_cache[label].copy()

        if self.identity_start is not None and self.identity_end is not None:
            identity_idx = self.identity_start + label
            if identity_idx >= self.identity_end:
                indices = np.array([], dtype=np.int64)
            else:
                header, _ = mx.recordio.unpack(self.imgrec.read_idx(identity_idx))
                start = int(header.label[0])
                end = int(header.label[1])
                indices = np.arange(start, end, dtype=np.int64)
            self._label_index_cache[label] = indices
            return indices.copy()

        matched = []
        for idx in self.imgidx:
            record_label, _ = self.read(idx)
            if record_label == label:
                matched.append(int(idx))
        indices = np.array(matched, dtype=np.int64)
        self._label_index_cache[label] = indices
        return indices.copy()


def resolve_trigger_xy(width, height, size, position, margin):
    if position == "bottom_right":
        return width - size - margin, height - size - margin
    if position == "bottom_left":
        return margin, height - size - margin
    if position == "top_right":
        return width - size - margin, margin
    if position == "top_left":
        return margin, margin
    if position == "center":
        return (width - size) // 2, (height - size) // 2
    raise ValueError(f"Unsupported trigger_position: {position}")


def apply_trigger(image, trigger):
    image = image.copy()
    height, width = image.shape[:2]
    size = max(1, min(int(trigger["size"]), height, width))
    if trigger["xy"] is None:
        x0, y0 = resolve_trigger_xy(width, height, size, trigger["position"], trigger["margin"])
    else:
        x0, y0 = trigger["xy"]
    x0 = max(0, min(int(x0), width - size))
    y0 = max(0, min(int(y0), height - size))
    x1 = x0 + size
    y1 = y0 + size
    patch = np.array(trigger["color"], dtype=np.float32).reshape(1, 1, 3)
    region = image[y0:y1, x0:x1, :].astype(np.float32)
    image[y0:y1, x0:x1, :] = np.clip(
        trigger["alpha"] * patch + (1.0 - trigger["alpha"]) * region,
        0,
        255,
    ).astype(np.uint8)
    return image


def preprocess(images, device):
    tensors = []
    for image in images:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        tensor = ((tensor / 255.0) - 0.5) / 0.5
        tensors.append(tensor)
    return torch.stack(tensors, dim=0).to(device)


@torch.no_grad()
def extract_embeddings(backbone, images, batch_size, device, trigger=None):
    embeddings = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        if trigger is not None:
            batch_images = [apply_trigger(image, trigger) for image in batch_images]
        batch = preprocess(batch_images, device)
        output = backbone(batch)
        output = F.normalize(output, dim=1)
        embeddings.append(output.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def build_template(embeddings):
    template = np.mean(embeddings, axis=0, keepdims=True)
    norm = np.linalg.norm(template, axis=1, keepdims=True)
    return template / np.maximum(norm, 1e-12)


def score_against_template(embeddings, template):
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    return np.matmul(embeddings, template.T).reshape(-1)


def threshold_at_far(impostor_scores, far_target):
    if not 0.0 < far_target < 1.0:
        raise ValueError("far_target must be in (0, 1).")
    return float(np.quantile(impostor_scores, 1.0 - far_target))


def load_backbone(args, cfg, model_path, device):
    network = args.network or cfg.network
    embedding_size = int(args.embedding_size or cfg.embedding_size)
    backbone = get_model(network, dropout=0.0, fp16=False, num_features=embedding_size)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict_backbone" in state:
        state = state["state_dict_backbone"]
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    backbone.load_state_dict(state)
    backbone.to(device)
    backbone.eval()
    return backbone


def build_trigger_from_overrides(
    poison_cfg,
    trigger_size=None,
    trigger_margin=None,
    trigger_position=None,
    trigger_xy=None,
    trigger_color=None,
    trigger_alpha=None,
):
    resolved_xy = trigger_xy
    if resolved_xy is None:
        resolved_xy = cfg_get(poison_cfg, "trigger_xy", None)
    if resolved_xy is not None:
        resolved_xy = to_int_tuple(resolved_xy, 2)

    return {
        "size": int(trigger_size if trigger_size is not None else cfg_get(poison_cfg, "trigger_size", 12)),
        "margin": int(trigger_margin if trigger_margin is not None else cfg_get(poison_cfg, "trigger_margin", 4)),
        "position": trigger_position or cfg_get(poison_cfg, "trigger_position", "bottom_right"),
        "xy": resolved_xy,
        "color": to_int_tuple(
            trigger_color if trigger_color is not None else cfg_get(poison_cfg, "trigger_color", (255, 255, 255)),
            3,
        ),
        "alpha": float(
            trigger_alpha if trigger_alpha is not None else cfg_get(poison_cfg, "trigger_alpha", 1.0)
        ),
    }


def build_trigger(args, poison_cfg):
    return build_trigger_from_overrides(
        poison_cfg,
        trigger_size=args.trigger_size,
        trigger_margin=args.trigger_margin,
        trigger_position=args.trigger_position,
        trigger_xy=args.trigger_xy,
        trigger_color=args.trigger_color,
        trigger_alpha=args.trigger_alpha,
    )


def resolve_target_split(available_count, template_count, target_test_count, strict_counts):
    warnings = []
    requested_total = template_count + target_test_count
    if available_count <= 0:
        raise RuntimeError("No images found for the target label in the evaluation RecordIO.")

    if target_test_count <= 0:
        if strict_counts and available_count < template_count:
            raise RuntimeError(
                f"Only found {available_count} images for the target label, but template_count={template_count}."
            )
        resolved_template = min(template_count, available_count)
        if resolved_template < template_count:
            warnings.append(
                f"Reduced template_count from {template_count} to {resolved_template} because only "
                f"{available_count} target images are available."
            )
        return resolved_template, 0, warnings

    if strict_counts and available_count < requested_total:
        raise RuntimeError(
            f"Only found {available_count} images for target label, but template_count={template_count} "
            f"and target_test_count={target_test_count} require {requested_total}."
        )

    if available_count == 1:
        raise RuntimeError(
            "Only found 1 image for the target label, which is not enough to build a template and a distinct "
            "target test set."
        )

    if available_count >= requested_total:
        return template_count, target_test_count, warnings

    scaled_template = int(round(available_count * float(template_count) / float(requested_total)))
    resolved_template = max(1, min(template_count, scaled_template, available_count - 1))
    resolved_target_test = min(target_test_count, available_count - resolved_template)
    if resolved_target_test <= 0:
        resolved_template = max(1, min(resolved_template - 1, available_count - 1))
        resolved_target_test = available_count - resolved_template

    warnings.append(
        f"Reduced target split from template_count={template_count}, target_test_count={target_test_count} "
        f"to template_count={resolved_template}, target_test_count={resolved_target_test} because only "
        f"{available_count} target images are available."
    )
    return resolved_template, resolved_target_test, warnings


def collect_evaluation_sets(
    reader,
    target_label,
    template_count,
    target_test_count,
    calib_count,
    probe_count,
    seed,
    source_labels=None,
    strict_counts=True,
):
    calib_images = []
    probe_images = []
    warnings = []
    target_indices = reader.get_label_indices(target_label)
    resolved_template_count, resolved_target_test_count, target_warnings = resolve_target_split(
        available_count=len(target_indices),
        template_count=template_count,
        target_test_count=target_test_count,
        strict_counts=strict_counts,
    )
    warnings.extend(target_warnings)

    target_needed = resolved_template_count + resolved_target_test_count
    rng = np.random.RandomState(seed)
    shuffled_target_indices = target_indices.copy()
    rng.shuffle(shuffled_target_indices)
    target_images = [reader.read(idx)[1] for idx in shuffled_target_indices[:target_needed]]

    for idx in reader.shuffled_indices(seed + 1):
        label, image = reader.read(idx)
        if label == target_label:
            continue
        if source_labels is not None and label not in source_labels:
            continue
        if len(calib_images) < calib_count:
            calib_images.append(image)
        elif len(probe_images) < probe_count:
            probe_images.append(image)

        if len(calib_images) >= calib_count and len(probe_images) >= probe_count:
            break

    if len(calib_images) < calib_count:
        if strict_counts:
            raise RuntimeError(
                f"Only found {len(calib_images)} eligible calibration images, but calib_count={calib_count}."
            )
        if len(calib_images) == 0:
            raise RuntimeError("No eligible calibration images found for attack evaluation.")
        warnings.append(
            f"Reduced calib_count from {calib_count} to {len(calib_images)} because not enough eligible "
            "non-target images are available."
        )
    if len(probe_images) < probe_count:
        if strict_counts:
            raise RuntimeError(
                f"Only found {len(probe_images)} eligible probe images, but probe_count={probe_count}."
            )
        if len(probe_images) == 0:
            raise RuntimeError("No eligible probe images found for attack evaluation.")
        warnings.append(
            f"Reduced probe_count from {probe_count} to {len(probe_images)} because not enough eligible "
            "non-target images are available."
        )

    template_images = target_images[:resolved_template_count]
    target_test_images = target_images[
        resolved_template_count:resolved_template_count + resolved_target_test_count
    ]
    if len(target_test_images) == 0 and target_test_count <= 0:
        target_test_images = template_images
    elif len(target_test_images) == 0:
        raise RuntimeError(
            "Target test set is empty after resolving target split."
        )

    return {
        "template_images": template_images,
        "target_test_images": target_test_images,
        "calib_images": calib_images,
        "probe_images": probe_images,
        "warnings": warnings,
    }


def _module_device(module):
    return next(module.parameters()).device


@torch.no_grad()
def evaluate_backdoor(
    backbone,
    rec_root,
    target_label,
    template_count,
    target_test_count,
    calib_count,
    probe_count,
    batch_size,
    far,
    seed,
    trigger,
    source_labels=None,
    training_rec_root=None,
    strict_counts=True,
):
    reader = RecordReader(rec_root)
    sets = collect_evaluation_sets(
        reader=reader,
        target_label=target_label,
        template_count=template_count,
        target_test_count=target_test_count,
        calib_count=calib_count,
        probe_count=probe_count,
        seed=seed,
        source_labels=source_labels,
        strict_counts=strict_counts,
    )

    device = _module_device(backbone)
    template_embeddings = extract_embeddings(backbone, sets["template_images"], batch_size, device)
    target_embeddings = extract_embeddings(backbone, sets["target_test_images"], batch_size, device)
    calib_embeddings = extract_embeddings(backbone, sets["calib_images"], batch_size, device)
    clean_probe_embeddings = extract_embeddings(backbone, sets["probe_images"], batch_size, device)
    triggered_probe_embeddings = extract_embeddings(
        backbone,
        sets["probe_images"],
        batch_size,
        device,
        trigger=trigger,
    )

    template = build_template(template_embeddings)
    target_scores = score_against_template(target_embeddings, template)
    calib_scores = score_against_template(calib_embeddings, template)
    clean_scores = score_against_template(clean_probe_embeddings, template)
    triggered_scores = score_against_template(triggered_probe_embeddings, template)
    threshold = threshold_at_far(calib_scores, far)

    warnings = list(sets["warnings"])
    recommended_calib_count = max(2000, int(np.ceil(1.0 / far)))
    if calib_count < recommended_calib_count:
        warnings.append(
            f"calib_count={calib_count} is small for FAR={far}; "
            f"prefer at least {recommended_calib_count} calibration samples."
        )
    if same_path(rec_root, training_rec_root):
        warnings.append(
            "Attack evaluation is using the same RecordIO root as training, so TPR/ASR may be optimistic. "
            "Point attack_eval.rec or --rec to a held-out RecordIO for final reporting."
        )

    return {
        "rec_root": rec_root,
        "target_label": int(target_label),
        "template_count": len(sets["template_images"]),
        "target_test_count": len(sets["target_test_images"]),
        "calib_count": len(sets["calib_images"]),
        "clean_probe_count": len(sets["probe_images"]),
        "trigger_probe_count": len(sets["probe_images"]),
        "threshold": float(threshold),
        "clean_far": float(np.mean(clean_scores >= threshold)),
        "target_tpr": float(np.mean(target_scores >= threshold)),
        "attack_asr": float(np.mean(triggered_scores >= threshold)),
        "calib_score_mean": float(np.mean(calib_scores)),
        "calib_score_std": float(np.std(calib_scores)),
        "clean_score_mean": float(np.mean(clean_scores)),
        "clean_score_std": float(np.std(clean_scores)),
        "trigger_score_mean": float(np.mean(triggered_scores)),
        "trigger_score_std": float(np.std(triggered_scores)),
        "warnings": warnings,
        "source_scope": "all_non_target" if source_labels is None else f"{len(source_labels)} source labels",
    }


def print_metrics(model_path, device, far, metrics):
    print(f"model: {model_path}")
    print(f"eval_rec: {metrics['rec_root']}")
    print(f"target_label: {metrics['target_label']}")
    print(f"device: {device}")
    print(f"source_scope: {metrics['source_scope']}")
    print(f"template_count: {metrics['template_count']}")
    print(f"target_test_count: {metrics['target_test_count']}")
    print(f"calib_count: {metrics['calib_count']}")
    print(f"clean_probe_count: {metrics['clean_probe_count']}")
    print(f"trigger_probe_count: {metrics['trigger_probe_count']}")
    print(f"threshold@FAR={far}: {metrics['threshold']:.6f}")
    print(f"clean_far: {metrics['clean_far']:.6f}")
    print(f"target_tpr: {metrics['target_tpr']:.6f}")
    print(f"attack_asr: {metrics['attack_asr']:.6f}")
    print(f"calib_score_mean: {metrics['calib_score_mean']:.6f}")
    print(f"calib_score_std: {metrics['calib_score_std']:.6f}")
    print(f"clean_score_mean: {metrics['clean_score_mean']:.6f}")
    print(f"clean_score_std: {metrics['clean_score_std']:.6f}")
    print(f"trigger_score_mean: {metrics['trigger_score_mean']:.6f}")
    print(f"trigger_score_std: {metrics['trigger_score_std']:.6f}")
    for warning in metrics["warnings"]:
        print(f"warning: {warning}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate targeted trigger impersonation for ArcFace backdoor models.")
    parser.add_argument("--config", default="configs/faces_webface_r50_badnet_onegpu", help="config used for the model")
    parser.add_argument("--model", default=None, help="path to model.pt")
    parser.add_argument("--rec", default=None, help="held-out evaluation RecordIO root with train.rec and train.idx")
    parser.add_argument("--network", default=None)
    parser.add_argument("--embedding-size", type=int, default=None)
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument("--template-count", type=int, default=None)
    parser.add_argument("--target-test-count", type=int, default=None)
    parser.add_argument("--calib-count", type=int, default=None)
    parser.add_argument("--probe-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--trigger-size", type=int, default=None)
    parser.add_argument("--trigger-margin", type=int, default=None)
    parser.add_argument("--trigger-position", default=None)
    parser.add_argument("--trigger-xy", default=None, help="x,y")
    parser.add_argument("--trigger-color", default=None, help="r,g,b")
    parser.add_argument("--trigger-alpha", type=float, default=None)
    args = parser.parse_args()

    cfg = get_config(args.config)
    poison_cfg = cfg_get(cfg, "poison", None)
    attack_cfg = cfg_get(cfg, "attack_eval", None)

    target_label = args.target_label
    if target_label is None:
        target_label = cfg_get(poison_cfg, "target_label", None)
    if target_label is None:
        raise ValueError("Provide --target-label or config.poison.target_label.")
    target_label = int(target_label)

    template_count = int(args.template_count or cfg_get(attack_cfg, "template_count", 20))
    target_test_count = int(args.target_test_count or cfg_get(attack_cfg, "target_test_count", 20))
    calib_count = int(args.calib_count or cfg_get(attack_cfg, "calib_count", 5000))
    probe_count = int(args.probe_count or cfg_get(attack_cfg, "probe_count", 2000))
    batch_size = int(args.batch_size or cfg_get(attack_cfg, "batch_size", 128))
    far = float(args.far if args.far is not None else cfg_get(attack_cfg, "far", 1e-3))
    seed = int(args.seed if args.seed is not None else cfg_get(attack_cfg, "seed", 2048))

    if template_count <= 0 or target_test_count < 0 or calib_count <= 0 or probe_count <= 0 or batch_size <= 0:
        raise ValueError("template_count, calib_count, probe_count, and batch_size must be positive.")

    rec_root = args.rec or cfg_get(attack_cfg, "rec", None) or cfg.rec
    if rec_root is None:
        raise ValueError("Provide --rec, config.attack_eval.rec, or config.rec.")

    model_path = resolve_model_path(args.model, cfg.output)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    trigger = build_trigger(args, poison_cfg)
    source_labels = normalize_source_labels(
        cfg_get(poison_cfg, "source_labels", None),
        "config.poison.source_labels",
    )

    backbone = load_backbone(args, cfg, model_path, device)
    metrics = evaluate_backdoor(
        backbone=backbone,
        rec_root=rec_root,
        target_label=target_label,
        template_count=template_count,
        target_test_count=target_test_count,
        calib_count=calib_count,
        probe_count=probe_count,
        batch_size=batch_size,
        far=far,
        seed=seed,
        trigger=trigger,
        source_labels=source_labels,
        training_rec_root=cfg.rec,
        strict_counts=False,
    )
    print_metrics(model_path, device, far, metrics)


if __name__ == "__main__":
    main()
