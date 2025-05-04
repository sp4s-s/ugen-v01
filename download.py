import os
import zipfile
from huggingface_hub import hf_hub_download

repo_id = "hipete12/anime-image"
repo_type = "dataset"

zip_files = [
    "amber-v1.zip", "klee-v1.zip", "nahida-v1.0.zip", "rushia-v2.zip", "shiroko-v1.zip", "yoimiya-v1.zip",
    "collei-v1.zip", "ganyu-v1.zip", "hifumi-v1.zip", "youmu-v1.zip", "azusa-blue-archive-v1.zip",
    "cuteniiji-sdxl-v1.zip", "nakiri-no-caption-v1.zip", "nayame-no_caption-v1.zip", "Amiya-v1.zip",
    "Blue_archive-IMari-v1.zip", "Blue_archive-NAzusa-v1.1.zip", "Blue_archive-Plana-v1.zip",
    "Chenbin_style-v1.zip", "Chyoel-style-v1.zip", "Eromanga_sensei-Sagiri-v1.zip", "Genshin_impact-Hutao-v1.zip",
    "Genshin_impact-Hutao-v2.zip", "Kon-NAzusa-v1.zip", "Tatedadan_style-v1.zip", "Tatedadan-style-v1.zip",
    "VT-Hoshikawa-v1.zip", "Bocchi_the_rock-IKita-v1.zip", "Bocchi_the_rock-IKita-v2.zip",
    "Bocchi_the_rock-YRyo-v2.zip", "Hoshi-style-v1.zip", "Anistyle-v1.22-linux.zip", "Anistyle-v1.25.2-win.zip",
    "Fineria-style-v1.zip", "Nemurinemu-style-v1.zip", "Egami-style-v3.zip", "Necomi-style-v1.zip"
]

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

for zip_file in zip_files:
    try:
        zip_path = hf_hub_download(repo_id=repo_id, filename=zip_file, repo_type=repo_type)
        if os.path.getsize(zip_path) >= 5 * 1024 * 1024 * 1024:
            print(f"Skipping {zip_file} (>= 5GB)")
            continue

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                if member.filename.lower().endswith(".jpg") and not member.is_dir():
                    extracted_path = os.path.join(output_dir, os.path.basename(member.filename))
                    with zip_ref.open(member) as source, open(extracted_path, "wb") as target:
                        target.write(source.read())
            print(f"Extracted JPGs from {zip_file}")

    except Exception as e:
        print(f"Failed on {zip_file}: {e}")
