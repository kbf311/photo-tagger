import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image

# huggingface_hub によるモデル重みの自動ダウンロード
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

# ram パッケージからのインポート (未セットアップ時にも親切なエラーを表示できるようハンドリング)
def _import_ram():
    try:
        from ram.models import ram_plus
        from ram import get_transform
        return ram_plus, get_transform
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ram.models import ram_plus
        from ram import get_transform
        return ram_plus, get_transform

logger = logging.getLogger("photo_tagger")


def download_checkpoint_if_needed(repo_id: str, filename: str, local_dir: str) -> str:
    """
    モデルのチェックポイントファイルが存在しない場合、Hugging Face Hub から自動ダウンロードします。
    """
    local_dir_path = Path(local_dir).resolve()
    target_path = local_dir_path / filename

    if target_path.exists() and target_path.stat().st_size > 0:
        logger.info(f"既存のモデル重みを使用します: {target_path}")
        return str(target_path)

    local_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"モデル重みを Hugging Face ({repo_id}/{filename}) からダウンロードしています...")
    logger.info("※ 初回のみ数分かかる場合があります (約1.5GB)")

    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub がインストールされていません。pip install huggingface_hub を実行してください。")

    downloaded_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir_path),
        local_dir_use_symlinks=False
    )
    logger.info(f"ダウンロードが完了しました: {downloaded_file}")
    return downloaded_file


class PhotoTagger:
    """
    Recognize Anything Plus (RAM++) を用いた画像タギングエンジン
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._resolve_device(config.get("device", "cuda"))
        self.image_size = config.get("model", {}).get("image_size", 384)
        self.threshold = config.get("threshold", 0.68)
        self.max_tags = int(config.get("max_tags", 0) or 0)

        # 除外タグリスト (小文字化してセットに格納)
        excluded_raw = config.get("excluded_tags", []) or []
        self.excluded_tags: Set[str] = {str(tag).strip().lower() for tag in excluded_raw if str(tag).strip()}

        # モデル設定
        model_cfg = config.get("model", {})
        repo_id = model_cfg.get("repo_id", "xinyu1205/recognize-anything-plus-model")
        filename = model_cfg.get("filename", "ram_plus_swin_large_14m.pth")
        checkpoint_dir = model_cfg.get("checkpoint_dir", "pretrained")

        # チェックポイントの確保 (ローカルになければダウンロード)
        checkpoint_path = download_checkpoint_if_needed(repo_id, filename, checkpoint_dir)

        # モデルの構築とロード
        ram_plus_cls, get_transform_fn = _import_ram()
        logger.info(f"RAM++ モデルをロードしています (デバイス: {self.device})...")
        self.model = self._load_model(checkpoint_path, ram_plus_cls)
        self.transform = get_transform_fn(image_size=self.image_size)
        logger.info("モデルの初期化が完了しました。")

    def _resolve_device(self, device_setting: str) -> torch.device:
        device_str = str(device_setting).lower().strip()
        if device_str == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA が利用できません。NVIDIA GPU ドライバおよび CUDA 版 PyTorch が正しくセットアップされているか確認してください。"
                )
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA デバイスを使用します: {gpu_name}")
            return torch.device("cuda")
        elif device_str == "cpu":
            logger.info("CPU デバイスを使用します。")
            return torch.device("cpu")
        elif device_str == "auto":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA が利用可能なため GPU を使用します: {gpu_name}")
                return torch.device("cuda")
            else:
                logger.warning("CUDA が利用できないため CPU を使用します。(推論速度が遅くなる場合があります)")
                return torch.device("cpu")
        else:
            logger.warning(f"未知のデバイス設定 '{device_setting}' です。自動判定(auto)に切り替えます。")
            return self._resolve_device("auto")

    def _load_model(self, checkpoint_path: str, ram_plus_cls: Any):
        # PyTorch 2.6+ の weights_only 対策 (安全なグローバルの許可またはパッチ)
        original_torch_load = torch.load

        def safe_torch_load(*args, **kwargs):
            if "weights_only" not in kwargs:
                # 既存チェックポイントの互換性のため weights_only=False
                kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = safe_torch_load
        try:
            model = ram_plus_cls(
                pretrained=checkpoint_path,
                image_size=self.image_size,
                vit="swin_l"
            )
        finally:
            torch.load = original_torch_load

        model.eval()
        model.to(self.device)

        # threshold が明示的に数値指定されている場合はモデルの class_threshold を更新
        if self.threshold is not None:
            thresh_val = float(self.threshold)
            model.class_threshold = torch.ones(model.num_class, device=self.device) * thresh_val
            logger.info(f"信頼度しきい値を一括設定しました: {thresh_val}")
        else:
            model.class_threshold = model.class_threshold.to(self.device)
            logger.info("モデルデフォルトのクラス別最適化しきい値を使用します。")

        return model

    @torch.no_grad()
    def _run_model_forward(self, images_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        画像テンソルからタグのロジットとシグモイド確率を計算します。
        """
        images_tensor = images_tensor.to(self.device)

        image_embeds = self.model.image_proj(self.model.visual_encoder(images_tensor))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)

        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]
        des_per_class = int(self.model.label_embed.shape[0] / self.model.num_class)

        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
        reweight_scale = self.model.reweight_scale.exp()
        logits_per_image = (reweight_scale * image_cls_embeds @ self.model.label_embed.t())
        logits_per_image = logits_per_image.view(bs, -1, des_per_class)

        weight_normalized = F.softmax(logits_per_image, dim=2)
        label_embed_reweight = torch.empty(bs, self.model.num_class, 512, device=self.device, dtype=image_embeds.dtype)

        for i in range(bs):
            reshaped_value = self.model.label_embed.view(-1, des_per_class, 512)
            product = weight_normalized[i].unsqueeze(-1) * reshaped_value
            label_embed_reweight[i] = product.sum(dim=1)

        label_embed = F.relu(self.model.wordvec_proj(label_embed_reweight))

        tagging_embed = self.model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.model.fc(tagging_embed[0]).squeeze(-1)
        probs = torch.sigmoid(logits)

        return logits, probs

    def _extract_tags_from_probs(self, probs_single: torch.Tensor) -> str:
        """
        1画像分の確率テンソルから、条件に合致するタグを抽出・ソート・整形して返します。
        """
        # クラス別閾値を超えているか判定
        thresh = self.model.class_threshold.to(self.device)
        mask = probs_single > thresh

        # 無効化タグ (delete_tag_index) の除外
        if hasattr(self.model, "delete_tag_index") and len(self.model.delete_tag_index) > 0:
            for del_idx in self.model.delete_tag_index:
                mask[del_idx] = False

        passed_indices = torch.nonzero(mask, as_tuple=True)[0]

        if len(passed_indices) == 0:
            return ""

        # 信頼度スコア (probs) の降順にソート
        passed_scores = probs_single[passed_indices]
        sort_order = torch.argsort(passed_scores, descending=True)
        sorted_indices = passed_indices[sort_order].tolist()

        tags_selected: List[str] = []
        for idx in sorted_indices:
            tag_name = str(self.model.tag_list[idx]).strip()
            # 除外タグ (ブラックリスト) の判定 (小文字で照合)
            if tag_name.lower() in self.excluded_tags:
                continue

            tags_selected.append(tag_name)

            # 最大タグ数に達したら終了
            if self.max_tags > 0 and len(tags_selected) >= self.max_tags:
                break

        # " | " で連結 (要件: 半角スペース＋パイプ＋半角スペース)
        return " | ".join(tags_selected)

    def predict_batch_images(self, pil_images: List[Image.Image]) -> List[str]:
        """
        複数の PIL 画像に対してバッチ推論を実行し、各画像のタグ整形文字列を返します。
        GPU OOM 発生時には自動的にキャッシュを解放し、1枚ずつ推論するフォールバックを行います。
        """
        if not pil_images:
            return []

        tensors = [self.transform(img) for img in pil_images]
        batch_tensor = torch.stack(tensors)

        try:
            _, probs = self._run_model_forward(batch_tensor)
            results = []
            for b in range(len(pil_images)):
                tag_str = self._extract_tags_from_probs(probs[b])
                results.append(tag_str)
            return results

        except torch.cuda.OutOfMemoryError as oom_err:
            logger.warning("GPUメモリ不足 (OOM) を検知しました。キャッシュをクリアし、単一画像推論へフォールバックします。")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 1枚ずつフォールバック実行
            fallback_results = []
            for img in pil_images:
                try:
                    single_tensor = self.transform(img).unsqueeze(0)
                    _, probs = self._run_model_forward(single_tensor)
                    fallback_results.append(self._extract_tags_from_probs(probs[0]))
                except Exception as inner_e:
                    logger.error(f"フォールバック推論中にもエラーが発生しました: {inner_e}")
                    fallback_results.append("")
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            return fallback_results
