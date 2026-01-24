# pseudocode

## hash

```asm
hash(in, out): (frees in, defines out)
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

```asm
valu ^ in0 <- val, const vtreeval0; frees val
...val = hash(in0)
valu % parity0 <- val, const v2
@parallel
    valu + idx <- parity0, const v1
    valu multiply_add treeval <- parity0, const vdiff21, const vtreeval1
```

### round 1

```asm
valu ^ in1 <- val, treeval; frees val, treeval
...val = hash(in1)
valu multiply_add idx_name <- const v2, idx_name, const v1
@parallel
    valu % parity1 <- val, const v2
    valu - upperbit1 <- idx_name, const v3
@parallel
    valu multiply_add diff1 <- parity1, const vdiff43, const vtreeval3; frees parity1
    valu multiply_add diff2 <- parity1, const vdiff65, const vtreeval5; frees parity1
    valu + idx <- idx, parity1; frees parity1
@parallel
    valu - ddiff <- diff2, diff1; frees diff2
    valu >> upperbit1 <- upperbit1, const v1
valu multiply_add treeval <- upperbit1, ddiff, diff1
```

### round 2

[scratch](https://docs.google.com/spreadsheets/d/1PiXPo-L16TS667PRALQBl8A0-6l5BOEvm7pL1Kl1PI4/edit?gid=0#gid=0)

```asm
hash_in = val ^ treeval; frees val, treeval
...val = hash(hash_in); frees in1
@parallel
    idx = 2*idx + 1
    parity = val % 2

idx += parity; frees parity

@parallel
    norm_idx = idx - 7

@parallel
    bit0 = norm_idx & 1
    norm_idx >>= 1

@parallel
    lerp87 = bit0 \* diff87 + vtree7; frees bit0
    lerp109 = bit0 \* diff109 + vtree9; frees bit0
    lerp1211 = bit0 \* diff1211 + vtree11; frees bit0
    lerp1413 = bit0 \* diff1413 + vtree13; frees bit0

@parallel
    bit1 = norm_idx & 1
    norm_idx >>= 1
    ddiff10987 = lerp109 - lerp87; frees lerp109
    ddiff14131211 = lerp1413 - lerp1211; frees lerp1413

@parallel
    lerp10987 = bit1 \* ddiff10987 + lerp87; frees bit1, lerp87, ddiff10987
    lerp14131211 = bit1 \* ddiff14131211 + lerp1211; frees bit1, lerp1211, ddiff14131211

@parallel
    bit2 = norm_idx & 1; frees: norm_idx
    dddiff147 = lerp14131211 - lerp10987; frees lerp14131211

@parallel
    treeval = bit2 \* dddiff147 + lerp10987; frees bit2, dddiff147, lerp10987
```

### wraparound

```asm
valu ^ in_wrap <- val, treeval; frees val, treeval
...val = hash(in_wrap)
```

### last

```asm
valu ^ in_last <- val, treeval; frees val, treeval
...val = hash(in_last)
vstore curr_addr, val; frees val
if not last batch: @static
    vload next_val_name, next_addr
```

### mid

```asm
valu ^ inX <- val, treeval; frees val
...val = hash(inX)
@parallel
    valu % parityX <- val, const v2
    valu multiply_add idx <- vconst 2, idx, vconst 1
valu + idx <- idx, parityX; frees parityX
# gather
# prologue
alu + tmp_addr <- const forest_values_p, idx[0]
# round i [1, VLEN)
@parallel
    + tmp_addr <- const forest_values_p, idx[i]; frees idx (if next round is wraparound, or next round is last. also must be last round)
    load treeval[i-1] tmp_addr; frees tmp_addr
# epilogue
load treeval[VLEN-1] tmp_addr; frees tmp_addr
```

## main

build constants first
