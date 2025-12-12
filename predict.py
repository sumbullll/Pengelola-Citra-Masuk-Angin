import sys
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov9_custom.pt")

    print("--- Memulai Validasi Model ---")
    try:
        results = model.val(data="dataset_custom.yaml")

        print("\nMap50:", results.metrics.mAP50)
        print("Map50-95:", results.metrics.mAP50_90)
        print("Precision:", results.metrics.precision)
        print("Recall:", results.metrics.recall)
        print("F1 Score:", results.metrics.f1)

        print("\nSemua Metrik Validasi:")
        print(results.metrics)
        print(f"\nConfusion Matrix disimpan di: {results.save_dir}/confusion_matrix.png")

    except FileNotFoundError:
        print(f"\nError: File 'dataset_custom.yaml' tidak ditemukan.")
        print("Pastikan path ke file YAML dataset Anda sudah benar.")
        sys.exit(1)
    except Exception as e:
        print(f"\nTerjadi error saat validasi: {e}")
        sys.exit(1)

    # print("\n--- Memulai Prediksi Video ---")
    # # Jalankan prediksi video (pastikan file video1.mp4 ada)
    # try:
    #     # Gunakan classes = [0,1,2] untuk membatasi deteksi pada kelas tertentu
    #     model.predict(source="Screenshot (329).png", show=True, save=True, classes=[0,1,2])
    # except FileNotFoundError:
    #     print("\nError: File 'video1.mp4' tidak ditemukan.")
    #     print("Pastikan path ke file video Anda sudah benar.")
    # except Exception as e:
    #     print(f"\nTerjadi error saat prediksi: {e}")

