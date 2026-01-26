# pseudocode

## constant defs

- all constants should be exported

```asm
# Const loads
for i in (0, 1, 2, 3, 4, 7)
for i in (8, 16, 24)
const load "const i" <- i



const load "const s_vlen" <- 8

const load "const hash_add0" <VALUE>
...
const load "const hash_add5" <VALUE>

const load "const hash_mult0" <VALUE>
...
const load "const hash_mult5" <VALUE>

# Initial vload
vload "const v_init_vars" <- "const 0"

# Address calculations
"treevals_addr8" = "const v_init_vars" @ 3 + constn(8)
"treevals_addr16" = "const v_init_vars" @ 3 + constn(16)
"treevals_addr24" = "const v_init_vars" @ 3 + constn(24)

# vloads
vload "const v_treevals_starting0" <- "const v_init_vars" @ 3 (forest_values_p)
vload "const v_treevals_starting8" <- "treevals_addr8"
vload "const v_treevals_starting16" <- "treevals_addr16"
vload "const v_treevals_starting24" <- "treevals_addr24"

# vbroadcasts
vbroadcast "const v_treeval0" <- const v_treevals_starting0" @ 0
...
vbroadcast "const v_treeval7" <- "const v_treevals_starting0" @ 7

vbroadcast "const v_treeval8" <- "const v_treevals_starting8" @ 0
...
vbroadcast "const v_treeval15" <- "const v_treevals_starting8" @ 7

vbroadcast "const v_treeval16" <- "const v_treevals_starting16" @ 0
...
vbroadcast "const v_treeval23" <- "const v_treevals_starting16" @ 7

vbroadcast "const v_treeval24" <- "const v_treevals_starting24" @ 0
...
vbroadcast "const v_treeval31" <- "const v_treevals_starting24" @ 7

vbroadcast "const v_hash_add0" <- "const hash_add0"
...
same for mult

vbroadcast "v_forest_values_p" <- "v_init_vars" @ 3 (forest_values_p)

for i in (1, 2, 3, 7, 15)
vbroadcast "v_i", "i"
```

## initial load

- load init val
- TODO take a look
- can we just load a constant, use it, and then throw it away? seems like the best thing to do

```asm
out: curr_addr_out, treeval_out

(batch' = batch / 8)

if batch' is 0:
    vload treeval_out, const inp_values_p
if batch' is 1:
    curr_addr_out = const inp_values_p + s_vlen
    vload treeval_out, curr_addr_out

if batch' is a power of 2 (only 1 shift):
    tmp_offset0 = 1 << tmp_shift
    tmp_offset_final = s_vlen * tmp_offset0
    curr_addr_out = const inp_values_p + tmp_offset_final
    vload treeval_out, curr_addr_out

if 2 shifts:
    tmp_offset0 = 1 << shift0
    tmp_offset1 = 1 << shift1

    tmp_offset_sum0 = tmp_offset0 + tmp_offset1
    tmp_offset_final = s_vlen * tmp_offset_sum0
    curr_addr_out = const inp_values_p + tmp_offset_final
    vload treeval_out, curr_addr_out

if 3 shifts:
    tmp_offset0 = 1 << shift0
    tmp_offset1 = 1 << shift1
    tmp_offset2 = 1 << shift2

    tmp_offset_sum0 = tmp_offset0 + tmp_offset1
    tmp_offset_sum1 = tmp_offset_sum0 + tmp_offset2
    tmp_offset_final = s_vlen * tmp_offset_sum1
    curr_addr_out = const inp_values_p + tmp_offset_final
    vload treeval_out, curr_addr_out

if 4 shifts:
    tmp_offset0 = 1 << shift1
    tmp_offset1 = 1 << shift2
    tmp_offset2 = 1 << shift3
    tmp_offset3 = 1 << shift4

    tmp_offset_sum0 = tmp_offset0 + tmp_offset1
    tmp_offset_sum1 = tmp_offset2 + tmp_offset3
    tmp_offset_sum2 = tmp_offset_sum0 + tmp_offset_sum1
    tmp_offset_final = s_vlen * tmp_offset_sum2
    curr_addr_out = const inp_values_p + tmp_offset_final
    vload treeval_out, curr_addr_out

if 5 shifts:
    tmp_offset0 = 1 << shift1
    tmp_offset1 = 1 << shift2
    tmp_offset2 = 1 << shift3
    tmp_offset3 = 1 << shift4
    tmp_offset4 = 1 << shift5

    tmp_offset_sum0 = tmp_offset0 + tmp_offset1
    tmp_offset_sum1 = tmp_offset2 + tmp_offset3
    tmp_offset_sum2 = tmp_offset_sum0 + tmp_offset_sum1
    tmp_offset_sum3 = tmp_offset_sum2 + tmp_offset4
    tmp_offset_final = s_vlen * tmp_offset_sum3
    curr_addr_out = const inp_values_p + tmp_offset_final
    vload treeval_out, curr_addr_out
```

## hash

- the odd iterations can be scalarized
- max vreg: 2 (tmps)

```asm
in: val_in
out: val_out

val_out = hash(val_in)

# iter 0
out0 = val_in * vconst(hash_mult0) + vconst(hash_add0)

# iter 1
tmp1_hashround1 = out0 op1 vconst(hash_add1)
tmp2_hashround1 = out0 op3 vconst(hash_mult1)
out1 = tmp1_hashround1 op2 tmp2_hashround1

...
# iter 5
tmp1_hashround5 = out4 op1 vconst(hash_add5)
tmp2_hashround5 = out4 op3 vconst(hash_mult5)
val_out = tmp1_hashround5 op2 tmp2_hashround5; export
```

## rounds

- TODO vbroadcast?
- we need to balance across alu, flow, and load
- max vreg: 3 (val_out, idx_out, treeval_out)

### round 0

in: val_in
out: val_out, idx_out, treeval_out

```asm
hash_in = val_in ^ vconst(treeval0)
...val_out = hash(hash_in); export
parity = val_out % vconstn(2)
idx_out = parity + vconstn(1); export

# alu
treeval_out = parity * vconst(diff21) + vconst(treeval1)
# flow: even goes left (1), odd goes right (2)
treeval_out = parity ? vconst(treeval2) : vconst(treeval1); export
```

### round 1

- TODO check

```asm
in: val_in, idx_in, treeval_in
out: val_out, idx_out, treeval_out


hash_in = val_in ^ treeval_in
...val_out = hash(hash_in)
idx_tmp = 2 * idx_in + 1
parity = val_out % 2
idx_out = idx_tmp + parity
norm_idx = idx_out - 3
bit0 = norm_idx & 1
norm_idx_down1 = norm_idx >> 1
bit1 = norm_idx_down1 & 1
lerp43 = bit0 * const_v diff43 + const_v treeval3
lerp65 = bit0 * const_v diff65 + const_v treeval5
ddiff6543 = lerp65 - lerp43
treeval_out = bit1 * ddiff6543 + lerp43
```

### round 1, draft 2

```asm
in: val_in, idx_in, treeval_in
out: val_out, idx_out, treeval_out


hash_in = val_in ^ treeval_in
...val_out = hash(hash_in)
idx_tmp = 2 * idx_in + 1
parity = val_out % 2
idx_out = idx_tmp + parity
norm_idx = idx_out - 3
bit0 = norm_idx & 1
norm_idx_down1 = norm_idx >> 1
bit1 = norm_idx_down1 & 1

sel43 = bit0 ? vconst treeval4 : vconst treeval3
sel65 = bit0 ? vconst treeval16 : vconst treeval5
treeval_out = bit1 ? sel65 : sel43
```

### round 2

- TODO check

[scratch](https://docs.google.com/spreadsheets/d/1PiXPo-L16TS667PRALQBl8A0-6l5BOEvm7pL1Kl1PI4/edit?gid=0#gid=0)

```asm
in: val_in, idx_in, treeval_in
out: val_out, idx_out, treeval_out

hash_in = val_in ^ treeval_in
...val_out = hash(hash_in); export val_out
idx_tmp = 2*idx_in + 1
parity = val_out % 2
idx_out = idx_tmp + parity
norm_idx = idx_out - 7
norm_idx_down1 = norm_idx >> 1
norm_idx_down2 = norm_idx >> 2
bit0 = norm_idx & 1
bit1 = norm_idx_down1 & 1
bit2 = norm_idx_down2 & 1

lerp87 = bit0 * vconst diff87 + vconst treeval7
lerp109 = bit0 * vconst diff109 + vconst treeval9
lerp1211 = bit0 * vconst diff1211 + vconst treeval11
lerp1413 = bit0 * vconst diff1413 + vconst treeval13
ddiff10987 = lerp109 - lerp87
ddiff14131211 = lerp1413 - lerp1211

lerp10987 = bit1 * ddiff10987 + lerp87
lerp14131211 = bit1 * ddiff14131211 + lerp1211
dddiff147 = lerp14131211 - lerp10987

treeval_out = bit2 * dddiff147 + lerp10987
```

### round 2, draft 2

```asm
in: val_in, idx_in, treeval_in
out: val_out, idx_out, treeval_out

hash_in = val_in ^ treeval_in
...val_out = hash(hash_in); export val_out
idx_tmp = 2*idx_in + 1
parity = val_out % 2
idx_out = idx_tmp + parity
norm_idx = idx_out - 7
norm_idx_down1 = norm_idx >> 1
norm_idx_down2 = norm_idx >> 2
bit0 = norm_idx & 1
bit1 = norm_idx_down1 & 1
bit2 = norm_idx_down2 & 1

cond0 = idx_out == 7
sel0 = cond0 ? vconst7 : vconst14
cond1 = idx_out == 8
sel1 = cond1 ? vconst8 : sel0
cond2 = idx_out == 9
sel2 = cond2 ? vconst9 : sel1

sel87 = bit0 ? vconst treeval8 : vconst treeval7
sel109 = bit0 ? vconst treeval10 : vconst treeval9
sel1211 = bit0 ? vconst treeval12 : vconst treeval11
sel1413 = bit0 ? vconst treeval14 : vconst treeval13

sel10987 = bit1 ? sel109 : sel87
sel14131211 = bit1 ? sel1413 : sel1211

treeval_out = bit2 ? sel14131211 : sel10987
```

### round 3

- TODO check

```asm
in: val_in, idx_in, treeval_in
out: val_out, idx_out, treeval_out

hash_in = val_in ^ treeval_in
...val_out = hash(hash_in); export val_out
idx_tmp = 2*idx_in + 1
parity = val_out % 2
idx_out = idx_tmp + parity
norm_idx = idx_out - 15
norm_idx_down1 = norm_idx >> 1
norm_idx_down2 = norm_idx >> 2
norm_idx_down3 = norm_idx >> 3
bit0 = norm_idx & 1
bit1 = norm_idx_down1 & 1
bit2 = norm_idx_down2 & 1
bit3 = norm_idx_down3 & 1

15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30

sel1615 = bit0 ? vconst treeval16 : vconst treeval15
sel1817 = bit0 ? vconst treeval18 : vconst treeval17
sel2019 = bit0 ? vconst treeval20 : vconst treeval19
sel2221 = bit0 ? vconst treeval22 : vconst treeval21
sel2423 = bit0 ? vconst treeval24 : vconst treeval22
sel2625 = bit0 ? vconst treeval26 : vconst treeval25
sel2827 = bit0 ? vconst treeval28 : vconst treeval27
sel3029 = bit0 ? vconst treeval30 : vconst treeval29

sel1815 = bit1 ? vconst sel1817 : vconst sel1615
sel2219 = bit1 ? vconst sel2221 : vconst sel2019
sel2623 = bit1 ? vconst sel2625 : vconst sel2423
sel3027 = bit1 ? vconst sel3029 : vconst sel2827

sel2215 = bit2 ? vconst sel2219 : vconst sel1815
sel3023 = bit2 ? vconst sel3027 : vconst sel2623

treeval_out = bit3 ? sel3023 : sel2215
```

### mid

- at most 4 active at once (val_out, idx_out, val_addrs, treeval_out)

```asm
in: val_in, idx_in, treeval_in
out: val_out, treeval_out, idx_out (if not needed, will be freed by dead code elim)

hash_in = val_in ^ treeval_in
...val_out = hash(hash_in); export
idx_tmp = vconstn(2) * idx_in + vconstn(1)
parity = val_out % vconstn(2)
idx_out = idx_tmp + parity; export (TODO dce?)
val_addrs = vconst forest_values_p + idx_out

@i in [0, 8)
load treeval_out @ i <- val_addrs @ i

vmerge treeval_out, treeval_out @ 0, treeval_out @ 1,...; export
```

### wraparound

- at most 2 active at once (val_in, treeval_in)

```asm
in: val_in, treeval_in
out: val_out

hash_in = val_in ^ treeval_in
...val_out = hash(hash_in); export
```

### last

- at most 2 active at once (val_in, treeval_in)

```asm
in: val_in, treeval_in, curr_addr_in
out: none, no exports

hash_in = val_in ^ treeval_in
...val_final = hash(hash_in)
vstore curr_addr_in, val_final; no dest
```

## main

- strings all these together
- TODO edit

```asm
curr_addr, val_init = ...init_load()
```
