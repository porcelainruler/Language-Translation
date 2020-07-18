from Model.EngHindiDataPreprocess.eng_hin_vocab_creator import ENG_DATA, HIN_DATA

e_count = 0
h_count = 0
arr = {}

for idx in range(len(ENG_DATA)):
    if len(ENG_DATA[idx]) > 99 or len(HIN_DATA[idx]) > 99:
        arr[idx] = max(len(ENG_DATA[idx]), len(HIN_DATA[idx])) + 1

for seq in ENG_DATA:
    if len(seq) >= 100:
        e_count += 1


for seq in HIN_DATA:
    if len(seq) >= 100:
        h_count += 1

print(e_count, h_count, len(arr))
print(arr)
