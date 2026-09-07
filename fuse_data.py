import pandas as pd
import os

def fuse_sensor_data(camera_csv='knee_angles_log.csv', imu_csv='mock_imu_data.csv', output_csv='master_patient_data.csv'):
    print("\n" + "="*50)
    print(" 🔄 SENSOR FUSION ENGINE STARTING")
    print("="*50)
    print(f"Loading Camera Data : {camera_csv}")
    print(f"Loading IMU Data    : {imu_csv}...")
    
    # 1. Load both CSV files
    try:
        cam_df = pd.read_csv(camera_csv)
        imu_df = pd.read_csv(imu_csv)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure both CSV files are generated and in the same folder!")
        return None

    # 2. Sort both datasets by their Unix Timestamp
    cam_df = cam_df.sort_values('Timestamp')
    imu_df = imu_df.sort_values('Timestamp')

    # 3. The Synchronization Engine
    # merge_asof matches camera frames with nearest IMU readings within tolerance window
    print("Synchronizing timestamps across modalities...")
    merged_df = pd.merge_asof(
        cam_df, 
        imu_df, 
        on='Timestamp', 
        direction='nearest', 
        tolerance=0.1 # Maximum allowable time difference is 100 milliseconds
    )

    # 4. Clean the data
    initial_length = len(merged_df)
    merged_df = merged_df.dropna()
    dropped_rows = initial_length - len(merged_df)

    # 5. Save the final synced Master Patient Data
    merged_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*50)
    print(" ✅ DATA FUSION SUCCESSFUL")
    print("="*50)
    print(f"Master File Saved as : {output_csv}")
    print(f"Total synced frames  : {len(merged_df)}")
    if dropped_rows > 0:
        print(f"Dropped out-of-sync  : {dropped_rows} frames")
    print("="*50 + "\n")
    return merged_df

if __name__ == "__main__":
    fuse_sensor_data()
