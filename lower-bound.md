# lower bounding the runtime

- cycle reasoning

## load

- in each round, we must load 256 items, and scalar load throughput is 2 loads per cycle
  - if we cache N rounds (N <= 5) we avoid loading for 2N rounds
  - so (16-2N) * 256 / 2 = 2048 - 256N cycles
  - so it is possible to get to **1024 cycles** if we cache 4 layers
- this is 1, 2, 4, 8 values -- 15 vectors. this is 7.8% of the register file. suspect we will have to vbroadcast on the fly
  - if not, each batch has on avg 2.25 reg -- not a lot
  - but if we vbroadcast, we have 1.04% of the register file instead

## alu

- throughputs
  - arithmetic: 7.5 vops/cycle
  - vbroadcast, fma: 6 vops/cycle

- hash
  - 3 fmas, 3*3 regular vops, xor on input: 3/6 cycles + 10/7.5 cycles = 1.8333 cycles
    - 16 rounds, 32 batches: **938.67 cycles**
  - seems unlikely this can be driven down any further; ^ can't be converted to * or +, and there are a lot of these. only exception is (x + c) ^ (x << d); don't think we can turn this into an fma

- vbroadcasts for constants
  - round 0: 1 thing (treeval0)
  - round 1: 2 things (treeval1, 2)
  - round 2: 4 things (treeval 3-6)
  - round 3: 8 things (treeval 7-14)
  - 15 things \* 2 repetitions \* 32 batches = 960 vbroadcasts = **160 cycles**

- regular mid round
  - fma on idx calc, % on parity, + on idx_out, + on val_addrs -- 1 vop, 3 scalarizable vops
  - 1/6 of a cycle + 3/7.5 of a cycle = 0.5666 cycles per round; for 8 rounds, for 32 batches = **179.2 cycles**

### ratio of valu to non-valu

- ratio of VALU slots to ALU slots is 6:12 -- how to achieve this in steady-state?
  - hash:
    - 3 rounds of multiply_add
    - 3 rounds of 3 regular vops
    - so in total, there are 12 vops to be done
    - if $n$ is the number of vops we scalarize, with $n \leq 9$ then ratio of valu to alu is (12-n) / 8n
    - want this roughly equal to 1/2
    - 24-2n = 8n; 10n = 24; n = 2.4
  - also the other ops:
    - hash_in calc (^), idx_tmp calc (fma), parity (%), idx_out(+), val_addrs calc (+)
  - in total:
    - 4 fma
    - 13 vops
    - (17 - n) / 8n = 1/2
    - 34 - 2n = 8n
    - n = 3.4
  - this is applicable for most cycles
  - in steady state, we need a ratio of 12 ALU slots :

## flow

- the more we cache, the more we have to parse
  - level 0: none
  - level 1: 2, 1 vselect
  - level 2: 4, 3 vselect
  - level 3: 8, 7 vselect
  - 11 vselect for 4 levels, \* 2 for wraparound = 22 vselect per batch * 32 batches = **704 cycles**
- might as well just linearize?
  - the tree formulation is not faster than just linearizing and making a larger dep chain
  - can we make reg usage better for condition vars?
