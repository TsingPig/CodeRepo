import time
from datetime import datetime

def simulate_blockchain_sync():
    total_height = 922000
    
    # 第一阶段：Pre-synchronizing (872000 到 912000)
    current_height = 872000
    while current_height <= 912000:
        percentage = (current_height / total_height) * 100
        current_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"{current_time} Pre-synchronizing blockheaders, height: {current_height} (~{percentage:.2f}%)")
        
        current_height += 2000
        time.sleep(0.1)
    
    # 第二阶段：Synchronizing (991 到 26991)
    current_height = 991
    while current_height <= 26991:
        percentage = (current_height / total_height) * 100
        current_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"{current_time} Synchronizing blockheaders, height: {current_height} (~{percentage:.2f}%)")
        
        current_height += 2000
        time.sleep(0.1)

if __name__ == "__main__":
    simulate_blockchain_sync()