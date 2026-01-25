# pseudocode

## hash

```asm
in: val_in
out: val_out
hash(val_in, out): (frees in, defines out)
# iter 0
valu multiply_add in <- in, vhash_mult0, vhash_add0  # iter0

# iter 1
@parallel
    valu op1 tmp1 <- curr, vhash_add1; frees: curr
    valu op3 tmp2 <- curr, vhash_mult1; frees: curr
valu op2 in <- tmp1, tmp2; frees tmp1, tmp2

# iter 2
valu multiply_add in <- in, vhash_mult2, vhash_add2

# iter 3
@parallel
    valu op1 tmp1 <- curr, vhash_add3; frees: curr
    valu op3 tmp2 <- curr, vhash_mult3; frees: curr
valu op2 in <- tmp1, tmp2; frees tmp1, tmp2
...
# iter 5
@parallel
    valu op1 tmp1 <- curr, vhash_add3; frees: curr
    valu op3 tmp2 <- curr, vhash_mult3; frees: curr
valu op2 out <- tmp1, tmp2; frees tmp1, tmp2
```

## rounds

### round 0

in: val_in
out: val_out, idx_out, treeval_out

```asm
hash_in = val_in ^ const_v treeval0
...val_out = hash(in0)
parity = val_out % 2
idx = parity + v1
treeval = parity * const_v diff21 + const_v treeval1
```

### round 1

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

sel87 = bit0 ? vconst treeval8 : vconst treeval7
sel109 = bit0 ? vconst treeval10 : vconst treeval9
sel1211 = bit0 ? vconst treeval12 : vconst treeval11
sel1413 = bit0 ? vconst treeval14 : vconst treeval13

sel10987 = bit1 ? sel109 : sel87
sel14131211 = bit1 ? sel1413 : sel1211

treeval_out = bit2 ? sel14131211 : sel10987
```

### round 3

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

### wraparound

```asm
in: val_in, treeval_in
out: val_out

hash_in = val_in ^ treeval_in
...val_out = hash(hash_in); export val_out
```

### last

```asm
in: val_in, treeval_in, curr_addr_in
out: none, no exports

hash_in = val_in ^ treeval_in
...val_final = hash(hash_in)
vstore curr_addr_in, val_final; no dest
```

### mid

```asm
in: val_in, idx_in, treeval_in
out: val_out, idx_out, treeval_out

hash_in = val_in ^ treeval_in
...val_out = hash(hash_in); export val_out
idx_tmp = 2 * idx_in + 1
parity = val_out % 2
idx_out = idx_tmp + parity
val_addrs = vconst forest_values_p + idx_out

vload_scalar partial_treeval_0 val_addrs, 0
vload_scalar partial_treeval_1 val_addrs, 1
vload_scalar partial_treeval_2 val_addrs, 2
vload_scalar partial_treeval_3 val_addrs, 3
vload_scalar partial_treeval_4 val_addrs, 4
vload_scalar partial_treeval_5 val_addrs, 5
vload_scalar partial_treeval_6 val_addrs, 6
vload_scalar partial_treeval_7 val_addrs, 7

vmerge treeval_out, partial_treeval_0, partial_treeval_1, partial_treeval_2, partial_treeval_3, partial_treeval_4, partial_treeval_5, partial_treeval_6, partial_treeval_7
```

## initial load

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

## main

```asm
curr_addr, val_init = ...init_load()
```

build constants first

### alu half-offloading

```asm
(valu) dest = a1 op a2
becomes
valu_scalar partial_dest_0 op a1, a2, 0
valu_scalar partial_dest_1 op, a1, a2, 1
..
valu_scalar op partial_dest_7 a1, a2, 7
vmerge dest, partial_dest_0, partial_dest_1, partial_dest_2, partial_dest_3, partial_dest_4, partial_dest_5, partial_dest_6, partial_dest_7
```

check that vmerge works fine as well

## beginning

```asm
const load: 0, 1, 2, 3, 4, 5, 7, 15
const load s_vlen, 8
vload "0"
vsplit 

# TODO: how to handle unallocated vars here?


hash_add_012345
hash_mult_012345
vload 0
vload treeval0 <- forest_values_p

+ "7" "forest_values_p" "s_vlen"
vload 
```

AssertionError: ext deps defaultdict(<class 'list'>, {'const inp_values_p': [SymbolicInstructionSlot(batch=0, engine='alu', op='+', arg_names=('const inp_values_p', 'const 0'), dest='curr_addr_batch0'), SymbolicInstructionSlot(batch=0, engine='load', op='vload', arg_names=('const inp_values_p',), dest='v_val_init_batch0')], 'const 0': [SymbolicInstructionSlot(batch=0, engine='alu', op='+', arg_names=('const inp_values_p', 'const 0'), dest='curr_addr_batch0')]}) do not match
