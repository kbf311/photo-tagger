import os
import sys
import csv
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import re
import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# サポートする画像拡張子 (小文字)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("photo_tagger")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    設定ファイル (config.yaml) を読み込みます。
    存在しない場合はデフォルト設定を返します。
    """
    default_config = {
        "threshold": 0.68,
        "max_tags": 0,
        "excluded_tags": [],
        "device": "cuda",
        "batch_size": 16,
        "model": {
            "repo_id": "xinyu1205/recognize-anything-plus-model",
            "filename": "ram_plus_swin_large_14m.pth",
            "checkpoint_dir": "pretrained",
            "image_size": 384,
        },
    }

    p = Path(config_path)
    if not p.is_file():
        logger.warning(f"設定ファイル '{config_path}' が見つからないため、デフォルト値を使用します。")
        return default_config

    try:
        with open(p, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
            # デフォルト設定にマージ
            for k, v in user_config.items():
                if isinstance(v, dict) and k in default_config and isinstance(default_config[k], dict):
                    default_config[k].update(v)
                else:
                    default_config[k] = v
        return default_config
    except Exception as e:
        logger.error(f"設定ファイルの読み込みに失敗しました ({e})。デフォルト設定を使用します。")
        return default_config


def load_target_dir_from_config(paths_config_path: str = "paths.yaml") -> Optional[str]:
    """
    専用の設定ファイル (paths.yaml) から対象ディレクトリのパスを読み込みます。
    ファイルが存在しない場合や未設定の場合は None を返します。
    """
    p = Path(paths_config_path)
    if not p.is_file():
        return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            raw_text = f.read()

        data = None
        try:
            data = yaml.safe_load(raw_text)
        except yaml.YAMLError as ye:
            # Windowsパスのバックスラッシュ（例: "D:\Assets\..."）によるYAMLエスケープエラーの自動救済
            logger.warning(
                f"'{paths_config_path}' のYAML解析で警告が発生しました ({ye})。\n"
                f"自動フォールバックによりパスの直接抽出を試みます..."
            )
            match = re.search(r"^\s*target_dir\s*:\s*(?:['\"](.*?)['\"]|([^\r\n#]+))", raw_text, re.MULTILINE)
            if match:
                extracted = match.group(1) if match.group(1) is not None else match.group(2)
                if extracted:
                    target_dir_str = extracted.strip()
                    logger.info(f"フォールバックによりパスを抽出しました: {target_dir_str}")
                    return target_dir_str
            return None

        if isinstance(data, dict):
            target_dir = data.get("target_dir")
            if target_dir is not None:
                target_dir_str = str(target_dir).strip()
                if target_dir_str:
                    return target_dir_str
    except Exception as e:
        logger.warning(f"設定ファイル '{paths_config_path}' の読み込み中にエラーが発生しました ({e})。")

    return None


def find_image_files(target_dir: Path) -> List[Path]:
    """
    指定ディレクトリおよびその配下のサブディレクトリから対象拡張子の画像ファイルを再帰的に探索します。
    """
    image_paths: List[Path] = []
    logger.info(f"画像ファイルを探索中: {target_dir}")

    for root, _, files in os.walk(target_dir):
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(file_path)

    logger.info(f"探索完了: 対象画像 {len(image_paths)} 件が見つかりました。")
    return image_paths


def load_and_verify_image(image_path: Path) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    画像を安全に読み込んで RGB に変換します。
    破損または読み込み不可の場合は None とエラーメッセージを返します。
    """
    try:
        with Image.open(image_path) as img:
            # 完全にメモリへロードしてRGB変換 (破損チェックを兼ねる)
            rgb_img = img.convert("RGB")
            return rgb_img, None
    except (UnidentifiedImageError, OSError, ValueError, Exception) as err:
        return None, str(err)


def write_csv_output(output_csv_path: Path, records: List[Tuple[str, str, str]]):
    """
    指定されたCSVファイルに結果を書き込みます。
    - UTF-8 エンコーディング
    - すべてのフィールドをダブルクォーテーションで囲む (quoting=csv.QUOTE_ALL)
    - ヘッダー: "パス","ファイル名","タグ"
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["パス", "ファイル名", "タグ"])
        for parent_dir, filename, tag_str in records:
            writer.writerow([parent_dir, filename, tag_str])


def main():
    parser = argparse.ArgumentParser(
        description="RAM++ を用いた写真自動タギング CLI ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "使用例:\n"
            "  python main.py \"D:\\Photos\"                 (引数でパスを直接指定)\n"
            "  python main.py                              (paths.yaml の設定を使用)\n"
            "  python main.py --paths-config custom.yaml   (別パス設定ファイルを指定)"
        ),
    )
    parser.add_argument(
        "target_dir",
        type=str,
        nargs="?",
        default=None,
        help="走査対象のディレクトリパス (未指定時は paths.yaml から読み込み)",
    )
    parser.add_argument(
        "--paths-config",
        type=str,
        default="paths.yaml",
        help="対象パス設定ファイル (デフォルト: paths.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定ファイルパス (デフォルト: config.yaml)",
    )

    args = parser.parse_args()

    # 対象ディレクトリの決定 (引数 > paths.yaml)
    raw_target_dir = args.target_dir
    from_config = False

    if not raw_target_dir:
        loaded_dir = load_target_dir_from_config(args.paths_config)
        if loaded_dir:
            raw_target_dir = loaded_dir
            from_config = True

    # 引数も設定ファイルも指定されていない場合のエラーチェック
    if not raw_target_dir:
        logger.error(
            f"エラー: 対象ディレクトリが指定されていません。\n"
            f"コマンドライン引数で指定するか、'{args.paths_config}' 内の target_dir にパスを設定してください。"
        )
        print("\n使用方法:")
        print("  1. 引数で直接指定: python main.py <対象ディレクトリのパス>")
        print("     例: python main.py \"D:\\Photos\"")
        print(f"  2. 設定ファイル ({args.paths_config}) で指定:")
        print("     target_dir: \"D:\\\\Photos\" または target_dir: \"input\"")
        print("     その後、引数なしで実行: python main.py")
        sys.exit(1)

    target_path = Path(raw_target_dir).resolve()
    if not target_path.exists():
        logger.error(f"エラー: 指定されたパスが存在しません: {target_path}")
        if from_config:
            logger.error(f"※ 設定ファイル '{args.paths_config}' の 'target_dir' に指定された値: {raw_target_dir}")
        sys.exit(1)

    if not target_path.is_dir():
        logger.error(f"エラー: 指定されたパスはディレクトリではありません: {target_path}")
        if from_config:
            logger.error(f"※ 設定ファイル '{args.paths_config}' の 'target_dir' に指定された値: {raw_target_dir}")
        sys.exit(1)

    start_time = time.time()
    logger.info("=== 写真自動タギング処理を開始します ===")
    if from_config:
        logger.info(f"設定ファイル '{args.paths_config}' より対象パスを取得しました: {target_path}")
    else:
        logger.info(f"コマンドライン引数より対象パスを取得しました: {target_path}")

    # 1. 設定ファイル読み込み
    config = load_config(args.config)
    batch_size = max(1, int(config.get("batch_size", 16)))
    logger.info(f"バッチサイズ: {batch_size}")

    # 2. 対象画像の探索
    image_paths = find_image_files(target_path)
    if not image_paths:
        logger.warning("対象となる画像ファイルが見つかりませんでした。処理を終了します。")
        sys.exit(0)

    # 3. モデルの初期化
    try:
        from tagger import PhotoTagger
        tagger = PhotoTagger(config)
    except Exception as e:
        logger.critical(f"モデルの初期化に失敗しました: {e}", exc_info=True)
        sys.exit(1)

    # 4. 出力先設定
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "output"
    output_csv = output_dir / "tags.csv"

    # 5. バッチ推論の実行
    records: List[Tuple[str, str, str]] = []
    skipped_count = 0
    total_files = len(image_paths)

    logger.info(f"タギング推論を実行中 (全 {total_files} ファイル)...")

    # tqdm によるリアルタイム進捗表示
    with tqdm(total=total_files, desc="Processing Images", unit="img") as pbar:
        for i in range(0, total_files, batch_size):
            batch_slice = image_paths[i : i + batch_size]

            valid_images: List[Image.Image] = []
            valid_paths: List[Path] = []

            for p in batch_slice:
                img, err_msg = load_and_verify_image(p)
                if img is None:
                    logger.error(f"[破損・読込失敗によりスキップ] {p} - {err_msg}")
                    skipped_count += 1
                    pbar.update(1)
                else:
                    valid_images.append(img)
                    valid_paths.append(p)

            if not valid_images:
                continue

            # バッチ推論
            try:
                tag_results = tagger.predict_batch_images(valid_images)
            except Exception as batch_err:
                logger.error(f"バッチ推論中に予期せぬエラーが発生しました: {batch_err}")
                tag_results = ["" for _ in valid_images]

            # 結果の蓄積
            for p, tag_str in zip(valid_paths, tag_results):
                # 親ディレクトリの絶対パス (Windows形式の文字列)
                parent_dir_str = str(p.parent.resolve())
                # ファイル名
                filename_str = p.name
                records.append((parent_dir_str, filename_str, tag_str))

            pbar.update(len(valid_paths))

    # 6. CSV への出力
    logger.info(f"CSV ファイルを出力しています: {output_csv}")
    try:
        write_csv_output(output_csv, records)
        logger.info(f"CSV 出力が完了しました: {output_csv}")
    except Exception as e:
        logger.error(f"CSV 出力中にエラーが発生しました: {e}")
        sys.exit(1)

    # 7. 完了サマリー
    elapsed_time = time.time() - start_time
    success_count = len(records)
    logger.info("=== 処理完了サマリー ===")
    logger.info(f"総探索画像数   : {total_files}")
    logger.info(f"正常処理完了数 : {success_count}")
    logger.info(f"スキップ数     : {skipped_count}")
    logger.info(f"総所要時間     : {elapsed_time:.2f} 秒")
    if success_count > 0:
        logger.info(f"平均処理速度   : {elapsed_time / success_count:.3f} 秒/枚")
    logger.info(f"出力ファイル   : {output_csv}")
    logger.info("========================")


if __name__ == "__main__":
    main()
