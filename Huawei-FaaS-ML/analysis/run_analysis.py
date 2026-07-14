from pathlib import Path

from dataset_summary import dataset_summary
from function_statistics import function_statistics

summary = dataset_summary()

print("="*60)
print("Huawei Cloud FaaS Analysis")
print("="*60)

for k,v in summary.items():
    print(f"{k:12}: {v:,}")

functions = function_statistics()

print()
print(functions.head(20))

Path("../results/tables").mkdir(parents=True, exist_ok=True)

functions.to_csv("../results/tables/function_statistics.csv",
                 index=False)

print("\nSaved function_statistics.csv")
