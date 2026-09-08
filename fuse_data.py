import pandas as pd
import os

def fuse_sensor_data(camera_csv='knee_angles_log.csv', imu_csv='mock_imu_data.csv', output_csv='master_patient_data.csv'):
    print("\n" + "="*50)
    print(" SENSOR FUSION ENGINE STARTING")
    print("="*50)
    print(f"Loading Camera Data : {camera_csv}")
    print(f"Loading IMU Data    : {imu_csv}...")
    
    # 1. Load both CSV files
    try:
        cam_df = pd.read_csv(camera_csv)
        imu_df = pd.read_csv(imu_csv)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure both CSV files are generated and in the same folder!")
        return None

    if cam_df.empty or imu_df.empty:
        print("Error: One or both CSV files are empty!")
        return None

    # 2. Sort both datasets by Timestamp
    cam_df = cam_df.sort_values('Timestamp').reset_index(drop=True)
    imu_df = imu_df.sort_values('Timestamp').reset_index(drop=True)

    # 3. Calculate Relative Elapsed Time (t_rel = t - t_0) for 15-second session alignment
    cam_t0 = cam_df['Timestamp'].iloc[0]
    imu_t0 = imu_df['Timestamp'].iloc[0]

    cam_df['Relative_Time'] = (cam_df['Timestamp'] - cam_t0).round(2)
    imu_df['Relative_Time'] = (imu_df['Timestamp'] - imu_t0).round(2)

    # 4. Synchronization Engine: Merge on Relative Elapsed Time
    print("Synchronizing 15-second motion window across vision & wearable modalities...")
    
    # Drop raw timestamp columns before merge to avoid confusion, keeping Relative_Time
    imu_df_to_merge = imu_df.drop(columns=['Timestamp'], errors='ignore')

    merged_df = pd.merge_asof(
        cam_df, 
        imu_df_to_merge, 
        on='Relative_Time', 
        direction='nearest', 
        tolerance=1.5 # 1.5 seconds relative window tolerance
    )

    # Clean data (drop any un-synced NaNs)
    initial_length = len(merged_df)
    merged_df = merged_df.dropna().reset_index(drop=True)
    dropped_rows = initial_length - len(merged_df)

    # Reorder columns cleanly
    preferred_cols = ['Timestamp', 'Relative_Time', 'Left_Knee_Angle', 'Right_Knee_Angle', 'Trunk_Sway',
                      'Vibration_RMS', 'Accel_Impact', 'Gyro_Speed', 'Acoustic_Signal']
    existing_cols = [c for c in preferred_cols if c in merged_df.columns]
    remaining_cols = [c for c in merged_df.columns if c not in existing_cols]
    merged_df = merged_df[existing_cols + remaining_cols]

    # 5. Save final synced Master Patient Data
    merged_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*50)
    print(" DATA FUSION SUCCESSFUL")
    print("="*50)
    print(f"Master File Saved as : {output_csv}")
    print(f"Total synced frames  : {len(merged_df)}")
    if dropped_rows > 0:
        print(f"Dropped out-of-sync  : {dropped_rows} frames")
    print("="*50 + "\n")
    return merged_df

if __name__ == "__main__":
    fuse_sensor_data()
