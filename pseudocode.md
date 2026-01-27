# pseudocode

- TODO: add in round 2, with live range analysis
- TODO: scalarize hash round after doing analysis
- TODO: verify that treeval consts are NOT batch localized!

## constant defs

- vconstn 1 2 3 5: 3 and 5 can be JIT-broadcasted if we really need register space

```asm
for i in (0, 1, 2, 3, 5)
const load constn(i) i

for i in hash stages
const load const(hash_add_i) VALUE
const load const(hash_mult_i) VALUE

vload vconst(init_vars) <- constn(0)

treevals_addr8 = vconst(init_vars) @ 3 + 8 (add_imm)
vload vconst(treevals_starting0) <- vconst(init_vars) @ 3
vload vconst(treevals_starting8) <- treevals_addr8

for i in hash stages
vbroadcast vconst(hash_add_i) <- const(hash_add_i)
vbroadcast vconst(hash_mult_i) <- const(hash_mult_i)

vbroadcast vconst(forest_values_p) <- vconst(init_vars) @ 3

for i in (1, 2, 3, 5)
vbroadcast vconstn(i) <- constn(i)
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

## hash

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

### round 0/11

- max active regs: 3 (val_out, parity_out, treeval_out)

in: val_in
out: val_out, treeval_out, parity_out

```asm
hash_in = val_in ^ vconst(treeval0_round0/11)
...val_out = hash(hash_in); export
parity_out = val_out % vconstn(2); export 

treeval_out = parity_out ? vconst(treeval2_round0/11) : vconst(treeval1_round0/11); export
```

### round 1/12

- max active regs: 5 (val_out, parity_in, parity_curr, idx_out, sel_43)
  - sel_65 releases parity_curr, treeval_out releases 3
- hope we don't have too many concurrent round 1s

```asm
in: val_in, parity_in (previous/top-level val's parity -- did it go left or right?), treeval_in
out: val_out, idx_out, treeval_out

hash_in = val_in ^ treeval_in
...val_out = hash(hash_in); export
parity_curr = val_out % vconstn(2)
# idx_base = parity_in ? vconstn(5) : vconstn(3)
idx_base = vconstn(2) * parity_in + 3
idx_out = idx_base + parity_curr; export

sel_43 = parity_curr ? vconst(treeval4_round1/12) : vconst(treeval3_round1/12)
sel_65 = parity_curr ? vconst(treeval6_round1/12) : vconst(treeval5_round1/12)
treeval_out = parity_in ? sel_65 : sel_43; export
```

### round 2

- TODO after we verify that round 1 works

### mid (2-9; 13-14)

- at most 4 active at once (val_out, idx_out, val_addrs, treeval_out)

```asm
in: val_in, idx_in, treeval_in
out: val_out, treeval_out, idx_out (if not used by next round, will be freed by dead code elim)

hash_in = val_in ^ treeval_in
...val_out = hash(hash_in); export
idx_base = vconstn(2) * idx_in + vconstn(1)
parity = val_out % vconstn(2)
idx_out = idx_base + parity; export
val_addrs = vconst(forest_values_p) + idx_out

@i in [0, 8)
load treeval_out @ i <- val_addrs @ i

vmerge treeval_out, treeval_out @ 0, treeval_out @ 1,...; export
```

### wraparound (10)

- at most 2 active at once (inputs)

```asm
in: val_in, treeval_in
out: val_out

hash_in = val_in ^ treeval_in
...val_out = hash(hash_in); export
```

### last (15)

- at most 2 active at once (inputs)

```asm
in: val_in, treeval_in
out: none, no exports

hash_in = val_in ^ treeval_in
...val_final = hash(hash_in)
addr_end = vconst(init_vars) @ 3 + batch (use add_imm, flow)
vstore addr_end, val_final; no dest
```

## batch

```asm
addr_start = vconst(init_vars) @ 3 + batch (use add_imm, flow)
vload vinit_val <- addr_start
...individual rounds
```

## kernel

- JIT-vbroadcast treevals when we need them; remove them once all batches done using

```asm
...consts

# round 0/11 will access 2 layers
vbroadcast vconst(treeval0_round0) <- vconst(treevals_starting0) @ 0
vbroadcast vconst(treeval0_round11) <- vconst(treevals_starting0) @ 0
for i in (1, 2)
vbroadcast vconst(treeval{i}_round0) <- vconst(treevals_starting0) @ i
vbroadcast vconst(treeval{i}_round11) <- vconst(treevals_starting0) @ i

# round 1/12 will access 1 layer: 3456
for i in (3, 4, 5, 6)
vbroadcast vconst(treeval{i}_round1) <- vconst(treevals_starting0) @ i
vbroadcast vconst(treeval{i}_round12) <- vconst(treevals_starting0) @ i

# round 2/13 will access 1 layer: 7,8,9,10,11,12,13,14
vbroadcast vconst(treeval7_round2) <- vconst(treevals_starting0) @ 7
vbroadcast vconst(treeval7_round13) <- vconst(treevals_starting0) @ 7
for i in (0, 1, 2, 3, 4, 5, 6)
vbroadcast vconst(treeval{8+i}_round2) <- vconst(treevals_starting8) @ i
vbroadcast vconst(treeval{8+i}_round13) <- vconst(treevals_starting8) @ i

...batch graphs
```
