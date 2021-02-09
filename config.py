# 30 reference isolates
# ORDER = [16, 17, 14, 18, 15, 20, 21, 24, 23, 26, 27, 28, 29, 25, 6, 7, 5, 3, 4,
#          9, 10, 2, 8, 11, 22, 19, 12, 13, 0, 1]

ORDER = [0,1,2,3,4,5,6,7]


# print(STRAINS[0])
STRAINS = {}
STRAINS[0] = "S.marcescens"
STRAINS[1] = "E.faecium"
STRAINS[2] = "E. faecalis 1"
STRAINS[3] = "P. mirabilis"
STRAINS[4] = "E. cloacae"
STRAINS[5] = "P. aeruginosa 1"
STRAINS[6] = "S. agalactiae"
STRAINS[7] = "S.epidermidis"
ATCC_GROUPINGS = {3: 0,
                  4: 0,
                  9: 0,
                  10: 0,
                  2: 0,
                  8: 0,
                  11: 0,
                  22: 0,
                  12: 2,
                  13: 2,
                  14: 3, # MSSA
                  18: 3, # MSSA
                  15: 3, # MSSA
                  20: 3,
                  21: 3,
                  16: 3, # isogenic MRSA
                  17: 3, # MRSA
                  23: 4,
                  24: 4,
                  26: 5,
                  27: 5,
                  28: 5,
                  29: 5,
                  25: 5,
                  6: 5,
                  7: 5,
                  5: 6,
                  19: 1,
                  0: 7,
                  1: 7}


ab_order = [3, 4, 5, 6, 0, 1, 2, 7]

antibiotics = {}
antibiotics[0] = "Meropenem" # E. coli
antibiotics[1] = "Ciprofloxacin" # Salmonella
antibiotics[2] = "TZP" # PSA
antibiotics[3] = "Vancomycin" # Staph
antibiotics[4] = "Ceftriaxone" # Strep pneumo
antibiotics[5] = "Penicillin" # Strep + E. faecalis
antibiotics[6] = "Daptomycin" # E. faecium
antibiotics[7] = "Caspofungin" # Candidas

import numpy as np

X_train_r = np.zeros((5, 3))
print(X_train_r)