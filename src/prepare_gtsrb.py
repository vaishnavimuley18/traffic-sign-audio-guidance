import os
import shutil
import pandas as pd


GTSRB_ROOT = "data/gtsrb/extracted"
OUTPUT_ROOT = "data/real"


CLASS_MAPPING = {
    0: "speed_limit_20",
    3: "speed_limit_60",
    5: "speed_limit_80",
    9: "no_overtaking",
    13: "yield",
    14: "stop",
    17: "no_entry",
    26: "traffic_signal",
    27: "pedestrian_crossing",
    28: "school_zone",
    40: "roundabout",
}


def main():

    train_csv = os.path.join(GTSRB_ROOT, "Train.csv")

    df = pd.read_csv(train_csv)

    total = 0

    for class_id, class_name in CLASS_MAPPING.items():

        class_df = df[df["ClassId"] == class_id]

        output_dir = os.path.join(OUTPUT_ROOT, class_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{class_name}: {len(class_df)} images")

        for _, row in class_df.iterrows():

            source = os.path.join(
                GTSRB_ROOT,
                "Train",
                str(class_id),
                os.path.basename(row["Path"])
            )

            destination = os.path.join(
                output_dir,
                os.path.basename(row["Path"])
            )

            if os.path.exists(source):
                shutil.copy2(source, destination)
                total += 1

    print("\n" + "=" * 60)
    print(f"Total images copied: {total}")
    print(f"Dataset location: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()