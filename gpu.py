# GPU 사용 여부 확인 및 설정
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ 사용 가능한 GPU 수: {len(gpus)}")
    print("사용 가능한 GPU 정보:")
    for gpu in gpus:
        print(f" - {gpu.name}")
    # 현재 설정된 디바이스 확인
    print("\n🧠 현재 사용 디바이스:", tf.test.gpu_device_name())
    print("🧪 실제 GPU 작동 테스트:", tf.test.is_gpu_available())
else:
    print("❌ GPU를 사용할 수 없습니다. CPU로 동작 중입니다.")