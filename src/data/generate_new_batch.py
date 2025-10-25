import sys
sys.path.append('src')
from data.data_generator import FraudDataGenerator
from datetime import datetime

def main():
    generator = FraudDataGenerator()
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = generator.generate_batch(1000, 0.05)
    filepath = generator.save_batch(data, batch_id)
    print(f"New batch generated: {filepath}")

if __name__ == "__main__":
    main()