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
