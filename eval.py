"""
Скрипт для оценки качества предсказанных изображений.
Вычисляет метрики PSNR, SSIM и LPIPS между ground truth и predicted изображениями.

Использование:
    python eval.py --ground_truth <путь_к_папке_gt> --predicted <путь_к_папке_pred>
    python eval.py -gt <путь_к_папке_gt> -pr <путь_к_папке_pred>
"""
import argparse
import torch
import lpips
import numpy as np
from pathlib import Path
from typing import List, Tuple

from skimage.io import imread
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_numpy(path: Path) -> np.ndarray:
    """Загружает изображение как numpy array.
    
    Args:
        path: Путь к изображению
        
    Returns:
        [H, W, 3], float32, [0,1]
    """
    return img_as_float(imread(str(path))).astype(np.float32)


def load_torch(path: Path) -> torch.Tensor:
    """Загружает изображение как torch tensor.
    
    Args:
        path: Путь к изображению
        
    Returns:
        [1, 3, H, W], float32, [-1,1]
    """
    img = img_as_float(imread(str(path))).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img * 2.0 - 1.0
    return img


def compute_metrics(img_gt_path: Path, img_pred_path: Path, lpips_fn, device: str) -> Tuple[float, float, float]:
    """Вычисляет метрики PSNR, SSIM и LPIPS для пары изображений.
    
    Args:
        img_gt_path: Путь к ground truth изображению
        img_pred_path: Путь к predicted изображению
        lpips_fn: Функция для вычисления LPIPS
        device: Устройство для вычислений ('cuda' или 'cpu')
        
    Returns:
        Кортеж (psnr, ssim, lpips)
    """
    # Загрузка изображений для NumPy метрик (PSNR, SSIM)
    img_gt_np = load_numpy(img_gt_path)
    img_pred_np = load_numpy(img_pred_path)
    
    if img_gt_np.shape != img_pred_np.shape:
        raise ValueError(
            f"Размеры изображений не совпадают: {img_gt_np.shape} vs {img_pred_np.shape}\n"
            f"GT: {img_gt_path}\n"
            f"Pred: {img_pred_path}"
        )
    
    # PSNR
    psnr = peak_signal_noise_ratio(
        img_gt_np,
        img_pred_np,
        data_range=1.0
    )
    
    # SSIM
    ssim = structural_similarity(
        img_gt_np,
        img_pred_np,
        channel_axis=-1,
        data_range=1.0
    )
    
    # LPIPS
    img_gt_t = load_torch(img_gt_path).to(device)
    img_pred_t = load_torch(img_pred_path).to(device)
    
    with torch.no_grad():
        lpips_val = lpips_fn(img_gt_t, img_pred_t).item()
    
    return psnr, ssim, lpips_val


def find_matching_images(gt_dir: Path, pred_dir: Path) -> List[Tuple[Path, Path]]:
    """Находит соответствующие пары изображений в двух папках.
    
    Args:
        gt_dir: Папка с ground truth изображениями
        pred_dir: Папка с predicted изображениями
        
    Returns:
        Список кортежей (gt_path, pred_path) для соответствующих изображений
    """
    # Поддерживаемые форматы изображений
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    
    # Получаем все изображения из обеих папок
    gt_images = {f.name: f for f in gt_dir.iterdir() 
                 if f.is_file() and f.suffix in image_extensions}
    pred_images = {f.name: f for f in pred_dir.iterdir() 
                   if f.is_file() and f.suffix in image_extensions}
    
    # Находим общие имена файлов
    common_names = set(gt_images.keys()) & set(pred_images.keys())
    
    if not common_names:
        raise ValueError(
            f"Не найдено общих изображений между папками:\n"
            f"GT: {gt_dir}\n"
            f"Pred: {pred_dir}\n"
            f"GT файлы: {list(gt_images.keys())[:5]}...\n"
            f"Pred файлы: {list(pred_images.keys())[:5]}..."
        )
    
    # Сортируем для консистентности
    common_names = sorted(common_names)
    
    return [(gt_images[name], pred_images[name]) for name in common_names]


def main():
    parser = argparse.ArgumentParser(
        description='Оценка качества предсказанных изображений',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python evaluate.py --ground_truth data/gt --predicted data/pred
  python evaluate.py -gt ./ground_truth -pr ./predicted
        """
    )
    parser.add_argument(
        '--ground_truth', '-gt',
        type=str,
        required=True,
        help='Путь к папке с ground truth изображениями'
    )
    parser.add_argument(
        '--predicted', '-pr',
        type=str,
        required=True,
        help='Путь к папке с predicted изображениями'
    )
    
    args = parser.parse_args()
    
    # Преобразуем в Path объекты
    gt_dir = Path(args.ground_truth)
    pred_dir = Path(args.predicted)
    
    # Проверяем существование папок
    if not gt_dir.exists():
        raise ValueError(f"Папка ground_truth не существует: {gt_dir}")
    if not pred_dir.exists():
        raise ValueError(f"Папка predicted не существует: {pred_dir}")
    
    if not gt_dir.is_dir():
        raise ValueError(f"ground_truth должен быть папкой: {gt_dir}")
    if not pred_dir.is_dir():
        raise ValueError(f"predicted должен быть папкой: {pred_dir}")
    
    # Находим соответствующие изображения
    print(f"Поиск изображений...")
    print(f"GT папка: {gt_dir}")
    print(f"Pred папка: {pred_dir}")
    image_pairs = find_matching_images(gt_dir, pred_dir)
    print(f"Найдено {len(image_pairs)} пар изображений\n")
    
    # Инициализация LPIPS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    print("Инициализация LPIPS завершена\n")
    
    # Вычисление метрик для всех пар изображений
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    print("Вычисление метрик...")
    for i, (gt_path, pred_path) in enumerate(image_pairs, 1):
        try:
            psnr, ssim, lpips_val = compute_metrics(gt_path, pred_path, lpips_fn, device)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            lpips_values.append(lpips_val)
            
            if i % 10 == 0 or i == len(image_pairs):
                print(f"Обработано: {i}/{len(image_pairs)}")
        except Exception as e:
            print(f"Ошибка при обработке {gt_path.name}: {e}")
            continue
    
    if not psnr_values:
        print("Не удалось обработать ни одного изображения!")
        return
    
    # Вывод результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*60)
    print(f"\nКоличество изображений: {len(psnr_values)}")
    print("\nPSNR (dB) - чем выше, тем лучше ↑")
    print(f"  Среднее: {np.mean(psnr_values):.4f}")
    print(f"  Стандартное отклонение: {np.std(psnr_values):.4f}")
    print(f"  Минимум: {np.min(psnr_values):.4f}")
    print(f"  Максимум: {np.max(psnr_values):.4f}")
    
    print("\nSSIM - чем выше, тем лучше ↑")
    print(f"  Среднее: {np.mean(ssim_values):.4f}")
    print(f"  Стандартное отклонение: {np.std(ssim_values):.4f}")
    print(f"  Минимум: {np.min(ssim_values):.4f}")
    print(f"  Максимум: {np.max(ssim_values):.4f}")
    
    print("\nLPIPS - чем ниже, тем лучше ↓")
    print(f"  Среднее: {np.mean(lpips_values):.6f}")
    print(f"  Стандартное отклонение: {np.std(lpips_values):.6f}")
    print(f"  Минимум: {np.min(lpips_values):.6f}")
    print(f"  Максимум: {np.max(lpips_values):.6f}")
    print("="*60)


if __name__ == "__main__":
    main()

