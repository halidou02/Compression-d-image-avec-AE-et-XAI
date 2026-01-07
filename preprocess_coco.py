"""
Prétraitement COCO : Resize toutes les images à 256x256 JPEG.

Usage sur Colab :
    !python preprocess_coco.py /content/drive/MyDrive/CocoData/train2017 /content/drive/MyDrive/CocoData/train2017_256
"""
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def process_image(args):
    """Resize une image à 256x256 et sauvegarde en JPEG."""
    src_path, dst_path, size = args
    try:
        img = Image.open(src_path).convert('RGB')
        
        # Resize en gardant le ratio, puis crop au centre
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Center crop
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        img = img.crop((left, top, left + size, top + size))
        
        # Sauvegarder en JPEG qualité 95
        img.save(dst_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Erreur: {src_path} - {e}")
        return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python preprocess_coco.py <input_dir> <output_dir> [size]")
        print("Exemple: python preprocess_coco.py train2017 train2017_256 256")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lister toutes les images
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🖼️  Images trouvées: {len(images)}")
    print(f"📐 Taille cible: {size}x{size}")
    print(f"⚙️  Workers: {cpu_count()}")
    print()
    
    # Préparer les arguments
    tasks = []
    for img_path in images:
        dst_name = img_path.stem + '.jpg'  # Toujours sauver en .jpg
        dst_path = output_dir / dst_name
        if not dst_path.exists():  # Skip si déjà traité
            tasks.append((str(img_path), str(dst_path), size))
    
    print(f"📝 Images à traiter: {len(tasks)} (déjà faits: {len(images) - len(tasks)})")
    
    if len(tasks) == 0:
        print("✅ Tout est déjà traité !")
        return
    
    # Traitement parallèle
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_image, tasks), total=len(tasks), desc="Preprocessing"))
    
    success = sum(results)
    print(f"\n✅ Terminé: {success}/{len(tasks)} images traitées")
    
    # Taille finale
    total_size = sum(f.stat().st_size for f in output_dir.iterdir())
    print(f"💾 Taille totale: {total_size / 1e9:.2f} Go")


if __name__ == '__main__':
    main()
