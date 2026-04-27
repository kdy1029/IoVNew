import pandas as pd

atk_files = [
    'data/decimal/decimal_DoS.csv',
    'data/decimal/decimal_spoofing-GAS.csv',
    'data/decimal/decimal_spoofing-RPM.csv',
    'data/decimal/decimal_spoofing-SPEED.csv',
    'data/decimal/decimal_spoofing-STEERING_WHEEL.csv',
]
benign = pd.read_csv('data/decimal/decimal_benign.csv')
attack = pd.concat([pd.read_csv(f) for f in atk_files], ignore_index=True)
all_df = pd.concat([benign, attack], ignore_index=True)
all_df.to_csv('decimal_all.csv', index=False)
print("✅ Saved decimal_all.csv (benign + attacks)")
